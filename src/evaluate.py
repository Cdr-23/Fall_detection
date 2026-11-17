"""Score an out-of-fold prediction file with the shared metrics (see metrics.py).

Usage: python src/evaluate.py results/<algo>_oof.parquet [--tau 0.5] [--no-cost]

Writes results/<algo>_metrics.csv, one row per metric, columns:
  metric, value (pooled over the 5 folds), fold_mean, fold_std, fold_0 ... fold_4.
Unless --no-cost, also measures the latency of src/run_<algo>.py:predict_proba on one window
(ms, median of 100 calls) and the size of models/<algo>.* (KB), when they exist.
predict_proba receives rows of processed/features.parquet (window_id included: select your
columns by name, e.g. features.FEATURES) and the matching rows of processed/X_seq.npy.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import CFG, ROOT, check_oof, evaluate_by_fold, measure_latency, model_size_kb  # noqa: E402

PROCESSED = ROOT / "processed"


def load_predict_proba(algo):
    path = ROOT / "src" / f"run_{algo}.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"run_{algo}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "predict_proba", None)


def cost_rows(algo):
    rows = {}
    models = sorted((ROOT / "models").glob(f"{algo}.*"))
    if models:
        rows["model_size_kb"] = model_size_kb(models[0])
    fn = load_predict_proba(algo)
    if fn is not None:
        feats = pd.read_parquet(PROCESSED / "features.parquet")
        X = np.load(PROCESSED / "X_seq.npy", mmap_mode="r")
        rows["latency_ms_per_window"] = measure_latency(fn, feats, np.asarray(X))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oof")
    ap.add_argument("--tau", type=float, default=CFG["eval"]["tau"])
    ap.add_argument("--sequences", default=str(PROCESSED / "sequences.csv"))
    ap.add_argument("--meta", default=str(PROCESSED / "windows_meta.parquet"))
    ap.add_argument("--no-cost", action="store_true")
    args = ap.parse_args()

    path = Path(args.oof)
    algo = path.stem.removesuffix("_oof")
    meta = pd.read_parquet(args.meta) if Path(args.meta).exists() else None
    oof = check_oof(pd.read_parquet(path), meta)
    table = evaluate_by_fold(oof, pd.read_csv(args.sequences), args.tau)
    table.loc["tau", "value"] = args.tau
    if not args.no_cost:
        for k, v in cost_rows(algo).items():
            table.loc[k, "value"] = v
    out = path.with_name(f"{algo}_metrics.csv")
    table.to_csv(out)
    keys = ["roc_auc_fall", "pr_auc_fall", "f1_notfall", "f1_fall", "f1_lying", "event_recall",
            "delay_mean_s", "false_alerts_adl", "false_alerts_per_hour", "latency_ms_per_window",
            "model_size_kb"]
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(table.loc[[k for k in keys if k in table.index], ["value", "fold_mean", "fold_std"]])
    print(f"-> {out}")


if __name__ == "__main__":
    main()
