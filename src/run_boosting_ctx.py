"""Gradient boosting with context features (P7 experiment; see src/features_ctx.py).

Exactly the model of src/run_boosting.py (median imputation, balanced sample weights, same grid,
same inner tuning by sequence with tree_tuning.cross_fit / fit_final): only the columns change.
Three variants, fixed in advance:
  boosting_floor     26 base features + FLOOR (6)
  boosting_lookback  26 base features + LOOKBACK (14: 3-second look-back + previous windows)
  boosting_ctx       26 base features + the 20 context features  <- the candidate, chosen a priori
This is an experiment: it does not replace the retained model (boosting + confirmation).

Inputs: processed/windows_meta.parquet, processed/features_ctx.parquet (python src/features_ctx.py).
Outputs per variant: results/<algo>_oof.parquet, results/<algo>_params.csv, models/<algo>.joblib;
results/p7_cost.csv (latency per window on features_ctx.parquet, model size);
SHAP of boosting_ctx: results/p7_shap_summary.png, results/p7_shap_importance.csv.
Usage: python src/run_boosting_ctx.py [variant ...]      (~5 min for the three)
       python src/evaluate.py results/<algo>_oof.parquet --no-cost
"""
import functools
import sys

import joblib
import numpy as np
import pandas as pd

from common import MODELS, RESULTS, oof_frame, save_oof
from features import FEATURES
from features_ctx import FEATURES_CTX, FLOOR, LOOKBACK_FEATS
from metrics import measure_latency, model_size_kb
from preprocess import PROCESSED
from run_boosting import PARAM_GRID, explain, make_model
from tree_tuning import cross_fit, fit_final

VARIANTS = {"boosting_floor": FEATURES + FLOOR,
            "boosting_lookback": FEATURES + LOOKBACK_FEATS,
            "boosting_ctx": FEATURES_CTX}
CANDIDATE = "boosting_ctx"
_models = {}


def load_ctx(processed=PROCESSED):
    """windows_meta and features_ctx (same row order), and X_seq (unused by the model)."""
    meta = pd.read_parquet(processed / "windows_meta.parquet")
    feats = pd.read_parquet(processed / "features_ctx.parquet")
    assert (meta["window_id"].to_numpy() == feats["window_id"].to_numpy()).all()
    return meta, feats, np.load(processed / "X_seq.npy", mmap_mode="r")


def predict_proba(feats: pd.DataFrame, X_seq: np.ndarray, algo: str = CANDIDATE) -> np.ndarray:
    """(N, 3) probabilities of not fall / fall / lying from the final model of `algo`.
    `feats` must hold the context columns (features_ctx.parquet, or
    features_ctx.window_features_ctx for a new video)."""
    cols = VARIANTS[algo]
    missing = [c for c in cols if c not in feats.columns]
    if missing:
        raise KeyError(f"{algo} needs the context features of src/features_ctx.py; missing "
                       f"{len(missing)} columns (e.g. {missing[:3]}). Use "
                       "processed/<set>/features_ctx.parquet or features_ctx.window_features_ctx().")
    if algo not in _models:
        _models[algo] = joblib.load(MODELS / f"{algo}.joblib")
    return _models[algo].predict_proba(feats[cols])


def train(algo, meta, feats):
    X = feats[VARIANTS[algo]]
    print(f"== {algo}: {X.shape[1]} features", flush=True)
    proba, chosen = cross_fit(make_model(), PARAM_GRID, meta, X, weighted=True)
    print("->", save_oof(algo, oof_frame(meta, proba)))
    model, best = fit_final(make_model(), PARAM_GRID, meta, X, weighted=True)
    chosen = pd.concat([chosen, pd.DataFrame([{"fold": "final", **best}])], ignore_index=True)
    chosen.to_csv(RESULTS / f"{algo}_params.csv", index=False)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / f"{algo}.joblib")
    print("->", MODELS / f"{algo}.joblib", flush=True)
    return model


def main():
    algos = sys.argv[1:] or list(VARIANTS)
    meta, feats, X_seq = load_ctx()
    for algo in algos:
        model = train(algo, meta, feats)
        if algo == CANDIDATE:
            imp = explain(model, feats, VARIANTS[algo], prefix="p7_",
                          title="boosting_ctx, 26 base + 20 context features, all 70 sequences")
            print(imp.sort_values("shap_fall", ascending=False).head(10).round(3))
            print("->", RESULTS / "p7_shap_summary.png", RESULTS / "p7_shap_importance.csv")
    cost = []
    for algo in VARIANTS:
        if not (MODELS / f"{algo}.joblib").exists():
            continue
        fn = functools.partial(predict_proba, algo=algo)
        cost.append({"algo": algo, "n_features": len(VARIANTS[algo]),
                     "latency_ms_per_window": measure_latency(fn, feats, np.asarray(X_seq)),
                     "model_size_kb": model_size_kb(MODELS / f"{algo}.joblib")})
    cost = pd.DataFrame(cost)
    cost.to_csv(RESULTS / "p7_cost.csv", index=False)
    print(cost.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
