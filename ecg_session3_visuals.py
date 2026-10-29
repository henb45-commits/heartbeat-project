"""Prepare measured model weights and train-fitted PCA for the session-3 view."""
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
FEATURES = [f"T{i}" for i in range(1, 351)]


def prepare():
    folder = ROOT / "derived_session3"
    beats = pd.read_csv(folder / "beats.csv.gz", float_precision="round_trip")
    manifest = pd.read_csv(folder / "split_manifest.csv")
    split = beats.recording_id.map(manifest.set_index("key").split)
    train, test = beats[split == "train"], beats[split == "test"]
    assert set(train.recording_id).isdisjoint(test.recording_id)
    model = Pipeline([("scaler", StandardScaler()),
                      ("logistic", LogisticRegression(solver="lbfgs", C=1, max_iter=3000,
                                                       tol=1e-4, random_state=42))])
    with warnings.catch_warnings(), threadpool_limits(limits=2):
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(train[FEATURES], train.person_id)
        predicted = model.predict(test[FEATURES])
        saved = pd.read_csv(folder / "beat_predictions.csv")
        assert np.array_equal(test[["person_id","recording_id","beat_id"]].to_numpy(),
                              saved[["person_id","recording_id","beat_id"]].to_numpy())
        assert np.array_equal(predicted, saved.logistic.to_numpy()), "Model differs from recorded experiment"
        scaler, logistic = model.named_steps.values()
        pca = PCA(n_components=2, svd_solver="full")
        pca.fit(scaler.transform(train[FEATURES]))
        coordinates = pca.transform(scaler.transform(beats[FEATURES]))
    np.savez_compressed(folder / "visualization_model.npz", classes=logistic.classes_.astype(str),
                        coefficients=logistic.coef_, coordinates=coordinates,
                        explained_variance=pca.explained_variance_ratio_,
                        train_samples=np.array(len(train)),
                        source_digest=np.array(hashlib.sha256((folder / "beats.csv.gz").read_bytes()).hexdigest()))
    print(json.dumps({"model_predictions_match_saved":True, "train_beats":len(train),
                      "pca_variance":pca.explained_variance_ratio_.tolist(),
                      "iterations":logistic.n_iter_.tolist()}))


if __name__ == "__main__":
    prepare()
