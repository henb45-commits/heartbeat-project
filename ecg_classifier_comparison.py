"""Train both classifiers on the saved recording split; never modify source data."""
import hashlib
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from ecg_session3 import FEATURES, evaluate
from ecg_error_analysis import save_error_analysis

ROOT = Path(__file__).resolve().parent
SEED = 42


def run():
    source = ROOT / "derived_session3"
    output = ROOT / "derived_classifier_comparison"
    inputs = [source / "beats.csv.gz", source / "split_manifest.csv"]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    beats = pd.read_csv(inputs[0], float_precision="round_trip")
    manifest = pd.read_csv(inputs[1])
    assert not manifest.key.duplicated().any()
    assert not beats.duplicated(["recording_id", "beat_id"]).any()
    split = beats.recording_id.map(manifest.set_index("key")["split"])
    assert split.notna().all()
    assert set(split) <= {"train", "test", "excluded"}
    train, test = beats[split == "train"], beats[split == "test"]
    assert set(train.recording_id).isdisjoint(test.recording_id)
    assert set(test.person_id) <= set(train.person_id)
    for name, table in (("train", train), ("test", test)):
        assert set(table.recording_id) == set(manifest.loc[manifest["split"] == name, "key"])
    labels = sorted(train.person_id.unique())
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(
            solver="lbfgs", C=1, max_iter=3000, tol=1e-4, random_state=SEED)),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    }
    output.mkdir(exist_ok=True)
    summaries = []
    for name, model in models.items():
        with warnings.catch_warnings(), threadpool_limits(limits=2):
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(train[FEATURES], train.person_id)
            probabilities = model.predict_proba(test[FEATURES])
        classes = model.classes_
        predictions = test[["person_id", "recording_id", "beat_id"]].copy()
        predictions["predicted"] = classes[probabilities.argmax(axis=1)]
        # Both classifiers use identical mean-probability aggregation per recording.
        means = pd.DataFrame(probabilities, index=test.recording_id, columns=classes).groupby(level=0).mean()
        recording = test.groupby("recording_id").person_id.agg(lambda values: values.iloc[0]).to_frame()
        recording["predicted"] = classes[means.loc[recording.index].to_numpy().argmax(axis=1)]
        slug = name.lower().replace(" ", "_")
        for unit, frame in (("beat", predictions), ("recording", recording)):
            summary, report, cm, errors = evaluate(frame.person_id, frame.predicted, labels, name, unit)
            summaries.append(summary)
            frame.to_csv(output / f"{slug}_{unit}_predictions.csv", index=(unit == "recording"))
            report.to_csv(output / f"{slug}_{unit}_classification_report.csv")
            cm.to_csv(output / f"{slug}_{unit}_confusion.csv", index_label="true / predicted")
            errors.to_csv(output / f"{slug}_{unit}_errors.csv", index=False)
            save_error_analysis(output, slug, unit)
    metrics = pd.DataFrame(summaries)
    metrics.to_csv(output / "metrics.csv", index=False)
    for unit in ("recording", "beat"):
        comparison = metrics.loc[metrics.unit == unit, [
            "method", "accuracy", "macro_precision", "macro_recall", "macro_f1-score"]].copy()
        comparison.columns = ["Model", "Accuracy", "Precision", "Recall", "F1"]
        comparison.to_csv(output / f"comparison_{unit}.csv", index=False)
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    metadata = dict(seed=SEED, averaging="macro", zero_division=0,
                    recording_aggregation="mean predicted class probabilities",
                    input_sha256=hashes, train_recordings=train.recording_id.nunique(),
                    test_recordings=test.recording_id.nunique(), train_beats=len(train), test_beats=len(test),
                    python=platform.python_version(), sklearn=sklearn.__version__,
                    numpy=np.__version__, pandas=pd.__version__,
                    parameters={name: model.get_params(deep=False) if name == "Decision Tree"
                                else model.steps[-1][1].get_params() for name, model in models.items()})
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    run()
    print(pd.read_csv(ROOT / "derived_classifier_comparison/comparison_recording.csv").to_string(index=False))
