"""Gradient boosting WITHOUT class balancing (P3 variant of run_boosting.py).

Same pipeline, grid and inner tuning by sequence as run_boosting.py, but trained with uniform
sample weights, so that p1 is a plain estimate of the fall probability (useful for P5's cost-based
choice of tau). Measured out-of-fold on URFD: mean p1 = 0.062 here vs 0.085 with balanced
weights (true fall rate 0.075), Brier score 0.042 vs 0.045: the balancing shifted p1 only
moderately. Fewer alerts at tau = 0.5, so compare the two variants on the whole
precision/recall curve, not at a single tau. Decided a priori (not by comparing OOF scores).

Outputs: results/boosting_unweighted_oof.parquet, results/boosting_unweighted_params.csv,
models/boosting_unweighted.joblib.
Usage: python src/run_boosting_unweighted.py
"""
import joblib
import numpy as np
import pandas as pd

from common import FEATURES, MODELS, RESULTS, load_data, oof_frame, save_oof
from run_boosting import PARAM_GRID, make_model
from tree_tuning import cross_fit, fit_final

ALGO = "boosting_unweighted"
MODEL_PATH = MODELS / f"{ALGO}.joblib"
_model = None


def predict_proba(feats: pd.DataFrame, X_seq: np.ndarray) -> np.ndarray:
    """(N, 3) probabilities of not fall / fall / lying, from the final model."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model.predict_proba(feats[FEATURES])


def main():
    meta, feats, _, _ = load_data()
    X = feats[FEATURES]
    proba, chosen = cross_fit(make_model(), PARAM_GRID, meta, X, weighted=False)
    print("->", save_oof(ALGO, oof_frame(meta, proba)))

    model, best = fit_final(make_model(), PARAM_GRID, meta, X, weighted=False)
    chosen = pd.concat([chosen, pd.DataFrame([{"fold": "final", **best}])], ignore_index=True)
    chosen.to_csv(RESULTS / f"{ALGO}_params.csv", index=False)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("->", MODEL_PATH)


if __name__ == "__main__":
    main()
