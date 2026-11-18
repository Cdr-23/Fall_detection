"""Shared evaluation metrics (CLAUDE.md section 7.0). Every algorithm is scored with these.

Input: an out-of-fold prediction table (results/<algo>_oof.parquet) with columns
  window_id, seq, fold, end_frame, y_true, p0, p1, p2, y_pred
and processed/sequences.csv (seq, kind, fall_start, ...), which gives the fall events.

Window level
  ROC-AUC and PR-AUC (average precision) of class 1 (fall) from p1; precision, recall, F1 per
  class from y_pred (argmax); 3x3 confusion matrix (rows = true, columns = predicted).
  alert_precision / alert_recall / alert_f1: the binary alert p1 >= tau against y_true == 1.

Event level (alert = a window with p1 >= tau)
  A real fall = the 30 frames of label 0 of a fall-* sequence, starting at frame s0.
  It is detected if a window whose LAST frame lies in [s0, s0 + 44] (the phase + 0.5 s) is
  alerted. Delay = (end frame of the first alerted window in that zone - s0) / fps, in seconds.
  A false alert = an alerted window outside these zones; alerted windows less than 60 frames
  (2 s) apart count as one. False alerts per hour = false alerts in adl-* sequences / monitored
  ADL time, where monitored time = number of evaluated ADL windows x stride / fps (the time
  during which the model actually issued a decision; ~0.07 h for the 40 URFD ADLs).
  With so little ADL time the per-hour rate is very imprecise: always report the raw count too.

Cost
  measure_latency(): median time of predict_proba on ONE window (ms), model_size_kb().
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, confusion_matrix, \
    precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "params.yaml").read_text())
OOF_COLUMNS = ["window_id", "seq", "fold", "end_frame", "y_true", "p0", "p1", "p2", "y_pred"]
CLASS_NAMES = {0: "notfall", 1: "fall", 2: "lying"}


def check_oof(oof, meta=None):
    """Raise if the OOF table breaks the contract; return it sorted by window_id."""
    missing = [c for c in OOF_COLUMNS if c not in oof]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if oof["window_id"].duplicated().any():
        raise ValueError("duplicated window_id")
    p = oof[["p0", "p1", "p2"]].to_numpy(dtype=float)
    if np.isnan(p).any() or (p < -1e-6).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-4):
        raise ValueError("p0, p1, p2 must be probabilities summing to 1")
    if meta is not None:
        m = meta.set_index("window_id").loc[oof["window_id"]]
        if not (m["y"].to_numpy() == oof["y_true"].to_numpy()).all():
            raise ValueError("y_true differs from windows_meta.y")
        if not (m["fold"].to_numpy() == oof["fold"].to_numpy()).all():
            raise ValueError("fold differs from windows_meta.fold")
        if len(oof) != len(meta):
            warnings.warn(f"{len(meta) - len(oof)} windows of windows_meta have no prediction")
    return oof.sort_values("window_id").reset_index(drop=True)


def window_metrics(oof, tau=CFG["eval"]["tau"]):
    y = oof["y_true"].to_numpy(dtype=int)
    yp = oof["y_pred"].to_numpy(dtype=int)
    p1 = oof["p1"].to_numpy(dtype=float)
    is_fall = (y == 1).astype(int)
    out = {}
    both = 0 < is_fall.sum() < len(is_fall)
    out["roc_auc_fall"] = roc_auc_score(is_fall, p1) if both else np.nan
    out["pr_auc_fall"] = average_precision_score(is_fall, p1) if both else np.nan
    prec, rec, f1, sup = precision_recall_fscore_support(y, yp, labels=[0, 1, 2], zero_division=0)
    for c, name in CLASS_NAMES.items():
        out[f"precision_{name}"], out[f"recall_{name}"], out[f"f1_{name}"] = prec[c], rec[c], f1[c]
        out[f"support_{name}"] = int(sup[c])
    out["f1_macro"] = float(np.mean(f1))
    out["accuracy"] = float((y == yp).mean())
    alert = p1 >= tau
    tp = int((alert & (y == 1)).sum())
    out["alert_precision"] = tp / alert.sum() if alert.sum() else 0.0
    out["alert_recall"] = tp / (y == 1).sum() if (y == 1).sum() else np.nan
    a, b = out["alert_precision"], out["alert_recall"]
    out["alert_f1"] = 2 * a * b / (a + b) if (a + b) > 0 else 0.0
    cm = confusion_matrix(y, yp, labels=[0, 1, 2])
    for i in range(3):
        for j in range(3):
            out[f"cm_true{i}_pred{j}"] = int(cm[i, j])
    return out


def fall_events(sequences, seqs=None):
    """(seq, s0) of every fall; restricted to `seqs` if given."""
    ev = sequences.dropna(subset=["fall_start"])[["seq", "fall_start"]]
    ev = ev.rename(columns={"fall_start": "s0"}).astype({"s0": int})
    return ev[ev["seq"].isin(seqs)] if seqs is not None else ev


def _count_clusters(frames, gap):
    """Number of groups when points closer than `gap` are merged (chained)."""
    if len(frames) == 0:
        return 0
    f = np.sort(np.asarray(frames))
    return int(1 + (np.diff(f) >= gap).sum())


def event_metrics(oof, sequences, tau=CFG["eval"]["tau"], tol=CFG["eval"]["event_tolerance"],
                  merge=CFG["eval"]["fa_merge"], fps=CFG["data"]["fps"],
                  stride=CFG["window"]["stride"]):
    alert = oof["p1"].to_numpy(dtype=float) >= tau
    end = oof["end_frame"].to_numpy(dtype=int)
    seq = oof["seq"].to_numpy()
    in_zone = np.zeros(len(oof), bool)
    detected, delays = [], []
    for s, s0 in fall_events(sequences, set(seq)).itertuples(index=False):
        zone = (seq == s) & (end >= s0) & (end <= s0 + tol)
        in_zone |= zone
        hit = zone & alert
        detected.append(hit.any())
        if hit.any():
            delays.append((end[hit].min() - s0) / fps)
    kind = pd.Series(seq).map(dict(zip(sequences["seq"], sequences["kind"]))).to_numpy()
    fa = alert & ~in_zone
    fa_adl = sum(_count_clusters(end[fa & (seq == s)], merge) for s in np.unique(seq[kind == "adl"]))
    fa_fall = sum(_count_clusters(end[fa & (seq == s)], merge) for s in np.unique(seq[kind == "fall"]))
    adl_hours = (kind == "adl").sum() * stride / fps / 3600
    n = len(detected)
    return {"n_falls": n, "n_detected": int(np.sum(detected)),
            "event_recall": np.mean(detected) if n else np.nan,
            "delay_mean_s": float(np.mean(delays)) if delays else np.nan,
            "delay_median_s": float(np.median(delays)) if delays else np.nan,
            "false_alerts_adl": fa_adl, "adl_hours": adl_hours,
            "false_alerts_per_hour": fa_adl / adl_hours if adl_hours > 0 else np.nan,
            "false_alerts_fall_seqs": fa_fall,
            "adl_seqs_with_false_alert": int(sum(
                (fa & (seq == s)).any() for s in np.unique(seq[kind == "adl"])))}


def wilson(k, n, z=1.96):
    """95 % Wilson interval of a proportion k / n (e.g. falls detected): (low, high)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    centre, half = (p + z * z / (2 * n)), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / (1 + z * z / n), (centre + half) / (1 + z * z / n))


def evaluate(oof, sequences, tau=CFG["eval"]["tau"]):
    """All window and event metrics of one OOF table (dict)."""
    return {**window_metrics(oof, tau), **event_metrics(oof, sequences, tau)}


def evaluate_by_fold(oof, sequences, tau=CFG["eval"]["tau"]):
    """DataFrame indexed by metric: value (pooled over all folds), fold_mean, fold_std
    (std with ddof=1), and one column per fold."""
    pooled = pd.Series(evaluate(oof, sequences, tau), name="value")
    folds = {f"fold_{k}": pd.Series(evaluate(g, sequences, tau))
             for k, g in oof.groupby("fold")}
    per = pd.DataFrame(folds)
    out = pd.concat([pooled, per.mean(axis=1).rename("fold_mean"),
                     per.std(axis=1).rename("fold_std"), per], axis=1)
    out.index.name = "metric"
    return out


def measure_latency(predict_proba, feats, X_seq, n_calls=100, seed=CFG["seed"]):
    """Median wall time (ms) of predict_proba on a single window (real-time use)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(feats), n_calls)
    predict_proba(feats.iloc[[idx[0]]], X_seq[idx[:1]])       # warm-up
    times = []
    for i in idx:
        t = time.perf_counter()
        predict_proba(feats.iloc[[i]], X_seq[[i]])
        times.append(time.perf_counter() - t)
    return 1000 * float(np.median(times))


def model_size_kb(path):
    path = Path(path)
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1024
    return path.stat().st_size / 1024
