"""Shared metrics (CLAUDE.md §8). Every algorithm reports the same columns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CHUTE = 1  # class index of `chute`


def window_metrics(y: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Window-level metrics. Positive class = `chute`."""
    is_chute = (y == CHUTE).astype(int)
    score = proba[:, CHUTE]
    out = {
        "roc_auc": roc_auc_score(is_chute, score) if 0 < is_chute.sum() < len(is_chute) else np.nan,
        "pr_auc": average_precision_score(is_chute, score) if is_chute.sum() > 0 else np.nan,
        "recall_chute": recall_score(is_chute, pred == CHUTE, zero_division=0),
        "precision_chute": precision_score(is_chute, pred == CHUTE, zero_division=0),
        "f1_chute": f1_score(is_chute, pred == CHUTE, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
    }
    return out


def count_alert_runs(pred_seq: np.ndarray) -> int:
    """Number of contiguous runs of `chute` predictions (one run = one alert)."""
    is_alert = (pred_seq == CHUTE).astype(int)
    return int(np.sum(np.diff(np.concatenate([[0], is_alert])) == 1))


def event_metrics(df: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    """Event-level metrics for one test set.

    df needs: seq, is_fall, y, end, fps, n_frames (one row per window, sorted by end
    within each sequence). pred is aligned with df.
    """
    d = df.assign(pred=pred)
    falls = d[d["is_fall"]]
    adls = d[~d["is_fall"]]

    detected, delays = 0, []
    for _, g in falls.groupby("seq"):
        g = g.sort_values("end")
        true_idx = np.flatnonzero(g["y"].to_numpy() == CHUTE)
        pred_idx = np.flatnonzero(g["pred"].to_numpy() == CHUTE)
        if len(pred_idx):
            detected += 1
        if len(true_idx) and len(pred_idx):
            delays.append((g["end"].iloc[pred_idx[0]] - g["end"].iloc[true_idx[0]]) / g["fps"].iloc[0])

    n_alerts, adl_seconds = 0, 0.0
    for _, g in adls.groupby("seq"):
        g = g.sort_values("end")
        n_alerts += count_alert_runs(g["pred"].to_numpy())
        adl_seconds += g["n_frames"].iloc[0] / g["fps"].iloc[0]

    n_falls = falls["seq"].nunique()
    adl_hours = adl_seconds / 3600
    return {
        "event_recall": detected / n_falls if n_falls else np.nan,
        "delay_s_mean": float(np.mean(delays)) if delays else np.nan,
        "false_alerts_per_h": n_alerts / adl_hours if adl_hours > 0 else np.nan,
        "n_false_alerts": n_alerts,
        "adl_seconds": adl_seconds,
    }
