"""Threshold rules (P2, CLAUDE.md section 7.2), ported from Alain's src/p2.py to the shared data.

Deterministic, no ML. Imitates a simple alarm rule:
  lying     if torso_angle_mean >= t_lie and head_height_mean <= h_lie   (checked first: lying
            windows also have large hip speeds from keypoint jitter, so they must not reach the
            fall rule)
  fall      else if hip_vy_max >= v_fall and torso_angle_max >= t_fall
  not fall  otherwise (also when a feature is NaN)
The 4 thresholds are chosen by grid search on the training folds only. Score: mean of F2 of `fall`
(recall counts more than precision) and F1 of `lying` (F2 alone leaves the lying branch unused).
Probabilities are 0 or 1.

Outputs: results/rules_oof.parquet, results/rules_thresholds.csv (per fold + final),
models/rules.joblib (thresholds tuned on the 70 sequences).
Usage: python src/run_rules.py
"""
import itertools

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, fbeta_score

from common import MODELS, RESULTS, load_data, oof_frame, save_oof

MODEL_PATH = MODELS / "rules.joblib"
GRID = {
    "v_fall": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],         # torso lengths / s, positive downwards
    "t_fall": [20, 30, 40, 50, 60, 70],                    # degrees from vertical
    "t_lie": [5, 10, 15, 20, 30, 40, 50, 60],              # degrees
    "h_lie": [-0.25, 0.0, 0.25, 0.5, 0.75, 1.0],           # torso lengths, head above hips
}
_model = None


def predict_class(feats, p):
    lying = ((feats["torso_angle_mean"] >= p["t_lie"]) & (feats["head_height_mean"] <= p["h_lie"])).to_numpy()
    fall = ((feats["hip_vy_max"] >= p["v_fall"]) & (feats["torso_angle_max"] >= p["t_fall"])).to_numpy()
    pred = np.zeros(len(feats), dtype=int)
    pred[fall] = 1
    pred[lying] = 2
    return pred


def score(y, pred):
    return 0.5 * fbeta_score(y == 1, pred == 1, beta=2, zero_division=0) + \
        0.5 * f1_score(y == 2, pred == 2, zero_division=0)


def fit(feats, y):
    best, best_s = None, -1.0
    for vals in itertools.product(*GRID.values()):
        p = dict(zip(GRID, vals))
        s = score(y, predict_class(feats, p))
        if s > best_s:
            best, best_s = p, s
    return best, best_s


def predict_proba(feats: pd.DataFrame, X_seq: np.ndarray) -> np.ndarray:
    """(N, 3) probabilities of not fall / fall / lying (0 or 1), from the final thresholds."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return np.eye(3)[predict_class(feats, _model)]


def main():
    meta, feats, _, _ = load_data()
    y = meta["y"].to_numpy()
    proba = np.zeros((len(meta), 3))
    rows = []
    for k in sorted(meta["fold"].unique()):
        tr, te = (meta["fold"] != k).to_numpy(), (meta["fold"] == k).to_numpy()
        p, s = fit(feats[tr], y[tr])
        proba[te] = np.eye(3)[predict_class(feats[te], p)]
        rows.append({"fold": k, **p, "train_score": round(s, 4)})
        print(f"fold {k}: {p}  train score {s:.3f}")
    print("->", save_oof("rules", oof_frame(meta, proba)))

    p, s = fit(feats, y)
    rows.append({"fold": "final", **p, "train_score": round(s, 4)})
    pd.DataFrame(rows).to_csv(RESULTS / "rules_thresholds.csv", index=False)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(p, MODEL_PATH)
    print(f"final: {p}\n-> {RESULTS / 'rules_thresholds.csv'}\n-> {MODEL_PATH}")


if __name__ == "__main__":
    main()
