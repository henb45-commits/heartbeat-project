"""Analyze saved test predictions and confusion matrices without retraining."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent


def analyze_errors(predictions, cm):
    """Return individual errors, directed counts and unordered pair counts."""
    assert list(cm.index) == list(cm.columns)
    assert set(predictions.person_id) <= set(cm.index)
    assert set(predictions.predicted) <= set(cm.columns)
    np.testing.assert_array_equal(cm.to_numpy(), confusion_matrix(
        predictions.person_id, predictions.predicted, labels=cm.index.tolist()))
    examples = predictions.loc[predictions.person_id != predictions.predicted].copy()
    examples = examples.rename(columns={"person_id": "true_person", "predicted": "predicted_person"})
    examples["error"] = examples.true_person + " -> " + examples.predicted_person
    examples = examples.sort_values(["true_person", "predicted_person", "recording_id"])
    directed, pairs = [], []
    for i, person in enumerate(cm.index):
        for j, other in enumerate(cm.columns):
            count = int(cm.iloc[i, j])
            if i != j and count:
                directed.append(dict(true_person=person, predicted_person=other,
                                     error=f"{person} -> {other}", count=count))
            if i < j:
                reverse = int(cm.iloc[j, i])
                if count + reverse:
                    pairs.append(dict(person_a=person, person_b=other,
                                      a_to_b=count, b_to_a=reverse, total=count + reverse))
    directed = pd.DataFrame(directed, columns=["true_person", "predicted_person", "error", "count"])
    directed = directed.sort_values(["count", "true_person", "predicted_person"], ascending=[False, True, True])
    pairs = pd.DataFrame(pairs, columns=["person_a", "person_b", "a_to_b", "b_to_a", "total"])
    pairs = pairs.sort_values(["total", "person_a", "person_b"], ascending=[False, True, True])
    assert directed["count"].sum() == pairs.total.sum() == len(examples)
    return examples, directed, pairs


def save_error_analysis(folder, slug, unit):
    folder = Path(folder)
    prefix = f"{slug}_{unit}"
    predictions = pd.read_csv(folder / f"{prefix}_predictions.csv")
    cm = pd.read_csv(folder / f"{prefix}_confusion.csv", index_col=0)
    tables = analyze_errors(predictions, cm)
    for name, table in zip(("error_examples", "directed_confusions", "confused_pairs"), tables):
        table.to_csv(folder / f"{prefix}_{name}.csv", index=False)
    return tables


def run():
    folder = ROOT / "derived_classifier_comparison"
    for slug in ("logistic_regression", "decision_tree"):
        for unit in ("recording", "beat"):
            save_error_analysis(folder, slug, unit)


if __name__ == "__main__":
    run()
