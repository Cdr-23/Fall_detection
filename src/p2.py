"""P2: threshold rules and logistic regression (CLAUDE.md §7).

Both models read the window table from src/prepare.py and output classes
0 = pas chute, 1 = chute, 2 = allongé, plus a probability per class.

- p2-rules: deterministic thresholds. Rules have no fitted weights, so the
  thresholds are chosen on the outer training folds only. The test fold is
  never used for choosing anything. Probabilities are one-hot (the rules
  cannot give a calibrated probability).
- p2-logreg: standardised logistic regression on a short list of key
  features. C is chosen with an inner grouped CV on the training folds.

Run from the repo root:  python src/p2.py
Outputs: results/p2-rules.csv, results/p2-logreg.csv (one row per fold),
         results/p2-rules_windows.csv, results/p2-logreg_windows.csv
"""

from __future__ import annotations

import io
import itertools
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from metrics import CHUTE, event_metrics, window_metrics
from prepare import CFG, ROOT, load_windows

RESULTS = ROOT / "results"

# ----------------------------------------------------------------- rules ----
RULE_GRID = {
    "v_thr": [0.25, 0.5, 1.0, 1.5, 2.0, 3.0],   # max downward hip speed in window (torso lengths / s)
    "t_thr": [0, 5, 10, 20, 30, 45, 60],               # max torso angle from vertical in window (deg)
    "t_lie": [0, 10, 20, 30, 40, 50, 60, 70, 80],      # torso angle at end of window (deg)
    "h_lie": [-1.5, -1.0, -0.5, 0.0, 0.2, 0.5, 0.8, 1.0],  # head height above hips at end (torso lengths)
}


def rules_predict(df: pd.DataFrame, p: dict) -> tuple[np.ndarray, np.ndarray]:
    chute = ((df["vy_max"] >= p["v_thr"]) & (df["torso_max"] >= p["t_thr"])).to_numpy()
    lying = (~chute) & (df["torso_last"] >= p["t_lie"]).to_numpy() & (df["head_h_last"] <= p["h_lie"]).to_numpy()
    pred = np.zeros(len(df), dtype=int)
    pred[lying] = 2
    pred[chute] = CHUTE
    return pred, np.eye(3)[pred]


def predict_rules(params: dict, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return rules_predict(df, params)


def fit_rules(train: pd.DataFrame) -> tuple[dict, dict]:
    keys = list(RULE_GRID)
    best, best_f1 = None, -1.0
    for combo in itertools.product(*RULE_GRID.values()):
        p = dict(zip(keys, combo))
        pred, _ = rules_predict(train, p)
        # 3-class macro F1: optimising chute F1 alone left the allonge branch degenerate.
        f1 = f1_score(train["y"], pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best, best_f1 = p, f1
    return best, {**best, "train_macro_f1": round(best_f1, 4)}


# ------------------------------------------------------------- logreg ----
LOGREG_FEATURES = [
    "vy_max", "vy_std", "ay_min",
    "torso_mean", "torso_max", "torso_last",
    "box_hw_mean", "head_h_min", "head_h_last",
]
C_GRID = [0.01, 0.1, 1.0, 10.0]


def _make_model(C: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=int(CFG["seed"])),
    )


def fit_logreg(train: pd.DataFrame) -> tuple[object, dict]:
    X = train[LOGREG_FEATURES].to_numpy()
    y = train["y"].to_numpy()
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=int(CFG["seed"]))
    splits = list(inner.split(X, y, groups=train["seq"]))

    best_C, best_f1 = None, -1.0
    for C in C_GRID:
        oof_pred = np.zeros(len(y), dtype=int)
        for tr_idx, va_idx in splits:
            m = _make_model(C).fit(X[tr_idx], y[tr_idx])
            oof_pred[va_idx] = m.predict(X[va_idx])
        f1 = f1_score(y == CHUTE, oof_pred == CHUTE, zero_division=0)
        if f1 > best_f1:
            best_C, best_f1 = C, f1

    model = _make_model(best_C).fit(X, y)
    assert list(model[-1].classes_) == [0, 1, 2], "every fold must contain all 3 classes"
    return model, {"C": best_C, "inner_f1_chute": round(best_f1, 4)}


def predict_logreg(model, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(df[LOGREG_FEATURES].to_numpy())
    return proba.argmax(axis=1), proba


# -------------------------------------------------------------- runner ----
def model_size_bytes(name: str, model, info: dict) -> int:
    if name == "p2-rules":
        return len(json.dumps(info).encode())
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return len(buf.getvalue())


def run() -> None:
    df = load_windows()
    RESULTS.mkdir(exist_ok=True)
    n_folds = int(CFG["n_folds"])

    specs = {
        "p2-rules": (fit_rules, predict_rules),
        "p2-logreg": (fit_logreg, predict_logreg),
    }
    for name, (fit, predict) in specs.items():
        rows, preds = [], []
        for k in range(n_folds):
            train, test = df[df["fold"] != k], df[df["fold"] == k]
            model, info = fit(train)

            t0 = time.perf_counter()
            pred, proba = predict(model, test)
            ms = (time.perf_counter() - t0) * 1000

            frames = test.drop_duplicates("seq")["n_frames"].sum()
            row = {
                "algo": name, "fold": k,
                **window_metrics(test["y"].to_numpy(), proba, pred),
                **event_metrics(test, pred),
                "infer_ms_per_frame": ms / frames,
                "model_size_bytes": model_size_bytes(name, model, info),
                "params": json.dumps(info),
            }
            rows.append(row)
            preds.append(pd.DataFrame({
                "algo": name, "fold": k, "seq": test["seq"].to_numpy(),
                "end": test["end"].to_numpy(), "y": test["y"].to_numpy(), "pred": pred,
                "p_pas_chute": proba[:, 0], "p_chute": proba[:, 1], "p_allonge": proba[:, 2],
            }))

        res = pd.DataFrame(rows)
        res.to_csv(RESULTS / f"{name}.csv", index=False)
        pd.concat(preds).to_csv(RESULTS / f"{name}_windows.csv", index=False)

        num = res.select_dtypes("number").drop(columns=["fold"])
        print(f"\n== {name} ({n_folds} folds, mean +/- std) ==")
        for col in num.columns:
            print(f"  {col:20s} {num[col].mean():10.4f} +/- {num[col].std():.4f}")
        print("  params per fold:", res["params"].tolist())


if __name__ == "__main__":
    run()
