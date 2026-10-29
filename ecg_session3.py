"""Reproducible recording-disjoint ECG-ID comparison; never writes source data."""
from pathlib import Path
import hashlib
import json
import platform
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ecg_baseline import SEED, analyze, normalize_template

ROOT = Path(__file__).resolve().parent
FEATURES = [f"T{i}" for i in range(1, 351)]


def source_hashes(data_root):
    return {str(p.relative_to(data_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(data_root.rglob("*")) if p.is_file()}


def beat_table(analysis):
    """T values are the existing extractor's centered, per-beat standardized signal."""
    tables = []
    for key, item in analysis["beats"].items():
        if item["fs"] != 500 or item["beats"].shape[1] != 350:
            raise ValueError("Expected 500 Hz and exactly 350 samples per beat")
        frame = pd.DataFrame(item["beats"], columns=FEATURES)
        frame.insert(0, "beat_id", item["used_peaks"].astype(int))
        frame.insert(0, "recording_id", key)
        frame.insert(0, "person_id", key.split("/")[0])
        tables.append(frame)
    table = pd.concat(tables, ignore_index=True)
    assert not table.duplicated(["recording_id", "beat_id"]).any()
    assert np.isfinite(table[FEATURES].to_numpy()).all()
    return table


def evaluate(truth, predicted, labels, method, unit):
    report = classification_report(truth, predicted, labels=labels,
                                   output_dict=True, zero_division=0)
    correct = int(np.sum(np.asarray(truth) == np.asarray(predicted)))
    summary = dict(method=method, unit=unit, total=len(truth), correct=correct,
                   errors=len(truth)-correct, accuracy=correct/len(truth))
    for average in ("macro", "weighted"):
        for metric in ("precision", "recall", "f1-score"):
            summary[f"{average}_{metric}"] = report[f"{average} avg"][metric]
    cm = pd.DataFrame(confusion_matrix(truth, predicted, labels=labels),
                      index=labels, columns=labels)
    pairs = pd.DataFrame({"true": list(truth), "predicted": list(predicted)})
    errors = (pairs[pairs.true != pairs.predicted].groupby(["true", "predicted"])
              .size().reset_index(name="count")
              .sort_values(["count", "true", "predicted"], ascending=[False, True, True]))
    return summary, pd.DataFrame(report).T, cm, errors


def run(output_dir=None):
    data_root = ROOT / "ecg-id-database-1.0.0"
    output = Path(output_dir) if output_dir else ROOT / "derived_session3"
    if output.resolve() == data_root.resolve() or data_root.resolve() in output.resolve().parents:
        raise ValueError("Derived outputs must be outside the source dataset")
    before = source_hashes(data_root)
    analysis = analyze(data_root)
    table = beat_table(analysis)
    train_keys, test_keys = set(analysis["train"]), set(analysis["test"])
    assert not train_keys & test_keys
    train = table[table.recording_id.isin(train_keys)].copy()
    test = table[table.recording_id.isin(test_keys)].copy()
    assert set(test.person_id) <= set(train.person_id)
    assert set(train.recording_id) == train_keys
    assert set(test.recording_id) == test_keys
    assert set(train.person_id) == set(analysis["labels"])

    model = Pipeline([("scaler", StandardScaler()),
                      ("logistic", LogisticRegression(solver="lbfgs", C=1.0,
                                                       max_iter=3000, tol=1e-4,
                                                       random_state=SEED))])
    # Fail rather than publishing a result from an unconverged fit.
    with warnings.catch_warnings(), threadpool_limits(limits=2):
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(train[FEATURES], train.person_id)
        probabilities = model.predict_proba(test[FEATURES])
    assert model.named_steps["scaler"].n_samples_seen_ == len(train)
    np.testing.assert_allclose(model.named_steps["scaler"].mean_,
                               train[FEATURES].mean().to_numpy(), atol=1e-12)
    labels = analysis["labels"]
    classes = model.named_steps["logistic"].classes_
    beat_predictions = test[["person_id", "recording_id", "beat_id"]].copy()
    beat_predictions["logistic"] = classes[probabilities.argmax(axis=1)]
    # Same original training gallery, applied to individual normalized beats.
    vectors = np.vstack([normalize_template(b) for b in test[FEATURES].to_numpy()])
    beat_predictions["baseline"] = np.asarray(labels)[
        (vectors @ analysis["gallery_matrix"].T).argmax(axis=1)]

    # Keep original baseline recording predictions exactly; average LR beat
    # probabilities to produce one decision on each of the SAME test recordings.
    probability_table = pd.DataFrame(probabilities, columns=classes,
                                     index=test.recording_id.to_numpy())
    mean_probabilities = probability_table.groupby(level=0).mean()
    recording_predictions = analysis["results"][["recording", "true", "predicted"]].rename(
        columns={"recording": "recording_id", "true": "person_id", "predicted": "baseline"})
    recording_predictions["logistic"] = [classes[mean_probabilities.loc[k].to_numpy().argmax()]
                                        for k in recording_predictions.recording_id]
    assert set(recording_predictions.recording_id) == test_keys
    assert len(recording_predictions) == len(test_keys)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "beats.csv.gz", index=False, compression="gzip")
    manifest = analysis["recordings"][["key", "participant", "n_beats", "error"]].copy()
    manifest["split"] = manifest.key.map(lambda k: "train" if k in train_keys else
                                         "test" if k in test_keys else "excluded")
    manifest.to_csv(output / "split_manifest.csv", index=False)
    summaries, evaluations = [], {}
    for unit, predictions in (("recording", recording_predictions), ("beat", beat_predictions)):
        predictions.to_csv(output / f"{unit}_predictions.csv", index=False)
        for method in ("baseline", "logistic"):
            summary, report, cm, errors = evaluate(predictions.person_id, predictions[method],
                                                   labels, method, unit)
            summaries.append(summary)
            evaluations[(unit, method)] = (report, cm, errors)
            report.to_csv(output / f"{unit}_{method}_classification_report.csv")
            cm.to_csv(output / f"{unit}_{method}_confusion.csv", index_label="true / predicted")
            errors.to_csv(output / f"{unit}_{method}_errors.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "metrics.csv", index=False)
    assert before == source_hashes(data_root), "Source dataset changed during analysis"
    metadata = dict(seed=SEED, python=platform.python_version(), sklearn=sklearn.__version__,
                    numpy=np.__version__, pandas=pd.__version__,
                    train_recordings=len(train_keys), test_recordings=len(test_keys),
                    train_beats=len(train), test_beats=len(test), all_beats=len(table),
                    classes=len(labels), excluded_recordings=int((manifest.split == "excluded").sum()),
                    source_files_unchanged=True, source_sha256=before,
                    logistic_parameters=model.named_steps["logistic"].get_params(),
                    iterations=model.named_steps["logistic"].n_iter_.tolist())
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_html(output, summary, evaluations, metadata)
    return dict(summary=summary, evaluations=evaluations, metadata=metadata,
                beats=table, model=model, train=train, test=test,
                recording_predictions=recording_predictions, beat_predictions=beat_predictions)


def write_html(output, summary, evaluations, metadata):
    parts = ['<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8">',
             '<title>ECG — מפגש 3</title><style>body{font-family:Arial;margin:35px;}'
             'table{border-collapse:collapse;direction:ltr}td,th{padding:7px;border:1px solid #ddd}</style>',
             '<h1>השוואת זיהוי ECG — מפגש 3</h1>',
             '<p>אותו סט הקלטות בדיקה לשתי השיטות. Macro נותן משקל שווה לכל אדם; '
             'Weighted משקלל לפי מספר הדוגמאות. מדדים למחלקה ללא ניבויים מוגדרים כ־0.</p>',
             f'<p>אימון: {metadata["train_recordings"]} הקלטות, {metadata["train_beats"]} פעימות. '
             f'בדיקה: {metadata["test_recordings"]} הקלטות, {metadata["test_beats"]} פעימות. '
             f'{metadata["classes"]} אנשים; {metadata["excluded_recordings"]} הקלטות מחוץ לניסוי.</p>',
             summary.round(4).to_html(index=False),
             '<p>ברמת הקלטה: baseline המקורי מול ממוצע הסתברויות הפעימות של Logistic Regression. '
             'ברמת פעימה: אותה גלריית baseline מול ניבוי Logistic Regression לכל פעימה.</p>']
    first = True
    for (unit, method), (report, cm, errors) in evaluations.items():
        parts.append(f'<h2 dir="ltr">{unit} — {method}</h2>')
        fig = go.Figure(go.Heatmap(z=cm.to_numpy(), x=cm.columns.tolist(), y=cm.index.tolist(),
                                  colorscale="Blues", hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"))
        fig.update_layout(height=650, xaxis_title="Predicted", yaxis_title="True",
                          yaxis_autorange="reversed")
        parts.append(fig.to_html(full_html=False, include_plotlyjs=True if first else False))
        first = False
        parts.extend(['<h3>הטעויות השכיחות — עד 15 זוגות</h3>', errors.head(15).to_html(index=False),
                      '<details><summary>Precision / Recall / F1 לכל אדם ולממוצעים</summary>',
                      report.round(4).to_html(), '</details>'])
    parts.append('<p>אין ערבוב הקלטות בין אימון לבדיקה. StandardScaler והתבניות נלמדים מהאימון בלבד. '
                 'הסינון ואיתור הפעימות נעשים בכל הקלטה בנפרד; נעשה שימוש בהקלטה השלמה, ולכן זו '
                 'בדיקה לא מקוונת ולא ניסוי בזמן אמת. הפיצול אינו מחייב הפרדה בין תאריכים. '
                 'לא בוצע כוונון פרמטרים לפי סט הבדיקה. זהו זיהוי סגור של אנשים מוכרים, ולא זיהוי אדם זר.</p></html>')
    (output / "comparison.html").write_text('\n'.join(parts), encoding="utf-8")


if __name__ == "__main__":
    result = run()
    print(result["summary"].to_string(index=False))
    print("Saved local outputs to derived_session3; source data unchanged.")
