"""Naive baseline: always predicts class 0 (not fall). Every model must beat it.

Outputs results/naive_oof.parquet and models/naive.joblib (CLAUDE.md section 7.0).
Usage: python src/run_naive.py
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from common import FEATURES, MODELS, load_data, oof_frame, save_oof

MODEL_PATH = MODELS / "naive.joblib"
_model = None


def make_model():
    return DummyClassifier(strategy="constant", constant=0)


def predict_proba(feats: pd.DataFrame, X_seq: np.ndarray) -> np.ndarray:
    """(N, 3) probabilities of not fall / fall / lying, from the final model."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model.predict_proba(feats[FEATURES])


def main():
    meta, feats, _, _ = load_data()
    X, y = feats[FEATURES], meta["y"].to_numpy()
    proba = np.zeros((len(meta), 3))
    for k in sorted(meta["fold"].unique()):
        tr, te = (meta["fold"] != k).to_numpy(), (meta["fold"] == k).to_numpy()
        proba[te] = make_model().fit(X[tr], y[tr]).predict_proba(X[te])
    print("->", save_oof("naive", oof_frame(meta, proba)))
    MODELS.mkdir(exist_ok=True)
    joblib.dump(make_model().fit(X, y), MODEL_PATH)
    print("->", MODEL_PATH)


if __name__ == "__main__":
    main()
