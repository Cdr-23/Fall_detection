"""Small helpers shared by the run_<algo>.py scripts (optional, but they guarantee the contract).

    from common import load_data, oof_frame, save_oof
    meta, feats, X_seq, valid = load_data()
    proba = np.zeros((len(meta), 3))
    for k in range(5):
        tr, te = meta.fold != k, meta.fold == k
        ... fit on tr (tune hyper-parameters with an inner GroupKFold on meta.seq[tr]) ...
        proba[te] = model.predict_proba(...)
    save_oof("myalgo", oof_frame(meta, proba))       # -> results/myalgo_oof.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURES  # noqa: F401  (re-exported: the feature column names)
from preprocess import CFG, PROCESSED, ROOT

RESULTS, MODELS = ROOT / "results", ROOT / "models"
SEED = CFG["seed"]


def load_data(processed=PROCESSED):
    """windows_meta, features, X_seq (N,30,17,2), valid (N,30); same row order."""
    p = Path(processed)
    meta = pd.read_parquet(p / "windows_meta.parquet")
    feats = pd.read_parquet(p / "features.parquet")
    assert (meta["window_id"].to_numpy() == feats["window_id"].to_numpy()).all()
    return meta, feats, np.load(p / "X_seq.npy"), np.load(p / "valid.npy")


def oof_frame(meta, proba):
    """OOF table in the contract format from (N,3) probabilities aligned with meta."""
    proba = np.asarray(proba, dtype=float)
    assert proba.shape == (len(meta), 3), proba.shape
    proba = proba / proba.sum(axis=1, keepdims=True)
    return pd.DataFrame({"window_id": meta["window_id"].to_numpy(), "seq": meta["seq"].to_numpy(),
                         "fold": meta["fold"].to_numpy(), "end_frame": meta["end_frame"].to_numpy(),
                         "y_true": meta["y"].to_numpy(), "p0": proba[:, 0], "p1": proba[:, 1],
                         "p2": proba[:, 2], "y_pred": proba.argmax(axis=1)})


def save_oof(algo, oof):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{algo}_oof.parquet"
    oof.to_parquet(path, index=False)
    return path
