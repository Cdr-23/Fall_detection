"""Sanity checks of the shared metrics and of the processed/ files.

Usage: python tests/test_metrics.py     (plain asserts, no pytest needed)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from metrics import check_oof, evaluate, evaluate_by_fold, event_metrics  # noqa: E402

P = ROOT / "processed"
META = pd.read_parquet(P / "windows_meta.parquet")
SEQS = pd.read_csv(P / "sequences.csv")


def oof_from_proba(proba):
    proba = np.asarray(proba, dtype=float)
    return pd.DataFrame({"window_id": META.window_id, "seq": META.seq, "fold": META.fold,
                         "end_frame": META.end_frame, "y_true": META.y, "p0": proba[:, 0],
                         "p1": proba[:, 1], "p2": proba[:, 2], "y_pred": proba.argmax(1)})


def test_perfect():
    oof = check_oof(oof_from_proba(np.eye(3)[META.y]), META)
    m = evaluate(oof, SEQS)
    for k in ["roc_auc_fall", "pr_auc_fall", "accuracy", "f1_macro", "alert_precision",
              "alert_recall", "event_recall"] + [f"{a}_{c}" for a in ("precision", "recall", "f1")
                                                 for c in ("notfall", "fall", "lying")]:
        assert m[k] == 1, (k, m[k])
    assert m["false_alerts_adl"] == 0 and m["false_alerts_fall_seqs"] == 0
    # perfect delay = first fall window ending at or after s0 (late when the fall starts
    # before frame 30, since the first window ends at frame 30)
    ev = SEQS.dropna(subset=["fall_start"])
    expected = [(META[(META.seq == s) & (META.y == 1)].end_frame.min() - s0) / 30
                for s, s0 in zip(ev.seq, ev.fall_start)]
    assert m["n_falls"] == 30 and abs(m["delay_mean_s"] - np.mean(expected)) < 1e-9
    t = evaluate_by_fold(oof, SEQS)
    assert (t.loc["event_recall"].drop("fold_std") == 1).all()


def test_naive():
    m = evaluate(oof_from_proba(np.tile([1.0, 0, 0], (len(META), 1))), SEQS)
    assert m["roc_auc_fall"] == 0.5 and m["recall_fall"] == 0 and m["event_recall"] == 0
    assert m["false_alerts_adl"] == 0 and np.isnan(m["delay_mean_s"])
    assert abs(m["pr_auc_fall"] - (META.y == 1).mean()) < 1e-9


def test_events_synthetic():
    seqs = pd.DataFrame({"seq": ["fall-x", "adl-x"], "kind": ["fall", "adl"],
                         "fall_start": [100, np.nan]})
    ends = np.arange(30, 301, 5)
    rows = []
    for s in ("fall-x", "adl-x"):
        for e in ends:
            rows.append({"seq": s, "end_frame": e, "p1": 0.0})
    oof = pd.DataFrame(rows)
    # fall detected by the window ending at 110 -> delay 10/30 s; window at 145 is outside
    oof.loc[(oof.seq == "fall-x") & oof.end_frame.isin([110, 115, 144, 145]), "p1"] = 0.9
    # ADL alerts at 50, 100 (gap 50 < 60 -> merged) and 200 (new): 2 false alerts
    oof.loc[(oof.seq == "adl-x") & oof.end_frame.isin([50, 100, 200]), "p1"] = 0.9
    m = event_metrics(oof, seqs, tau=0.5)
    assert m["n_detected"] == 1 and abs(m["delay_mean_s"] - 10 / 30) < 1e-9
    assert m["false_alerts_fall_seqs"] == 1, m
    assert m["false_alerts_adl"] == 2, m
    assert abs(m["adl_hours"] - len(ends) * 5 / 30 / 3600) < 1e-12


def test_processed_files():
    X, V = np.load(P / "X_seq.npy"), np.load(P / "valid.npy")
    F = pd.read_parquet(P / "features.parquet")
    folds = pd.read_csv(P / "folds.csv")
    assert X.shape == (len(META), 30, 17, 2) and X.dtype == np.float32
    assert V.shape == (len(META), 30) and V.dtype == bool
    assert (F.window_id.to_numpy() == META.window_id.to_numpy()).all()
    assert np.allclose(V.mean(1), META.valid_frac) and np.allclose(F.valid_frac, META.valid_frac)
    assert not np.isnan(X).any()
    assert (META.groupby("seq").fold.nunique() == 1).all()
    assert folds.seq.is_unique and len(folds) == 70
    assert (folds.set_index("seq").fold.loc[META.seq].to_numpy() == META.fold.to_numpy()).all()
    assert all((META[META.fold == k].y == 1).any() for k in range(5))
    assert (META.end_frame - META.start_frame == 29).all()
    assert set(META.y) == {0, 1, 2}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
