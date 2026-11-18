"""Human-readable features per window, computed from the cleaned poses (see preprocess.py).

Per-frame signals (image frame: y points down; scale = median torso length of the sequence):
  torso_angle  angle between the torso (hips -> shoulders) and the vertical, in degrees:
               degrees(arctan2(|dx|, -dy)); 0 standing, 90 lying, > 90 head below hips.
  hip_vy       vertical speed of the mid-hips in torso lengths per second, positive downwards,
               from the pixel positions divided by the scale, smoothed over 3 frames (trailing
               mean, so it only uses the past).
  hip_ay       derivative of hip_vy (torso lengths / s^2).
  bbox_ratio   height / width of the YOLO box (by2 - by1) / (bx2 - bx1).
  head_height  (y of the hips - y of the head) / scale, positive when the head is above the
               hips; head = mean of the available keypoints 0..4.

Per window and per signal: mean, min, max, std, delta (last valid value - first valid value),
named <signal>_<agg> (e.g. torso_angle_max), plus valid_frac (share of frames with a real
detection). Windows with no usable frame keep NaN: each model imputes with the TRAIN median.

Output: processed/features.parquet (window_id + features), same row order as windows_meta.
Usage: python src/features.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from preprocess import CFG, PROCESSED, L_HIP, R_HIP, L_SH, R_SH, load_poses, midpoint, \
    prepare_sequence

SIGNALS = ["torso_angle", "hip_vy", "hip_ay", "bbox_ratio", "head_height"]
AGGS = ["mean", "min", "max", "std", "delta"]
FEATURES = [f"{s}_{a}" for s in SIGNALS for a in AGGS] + ["valid_frac"]


def frame_signals(prep, fps=CFG["data"]["fps"]):
    """Per-frame signals of one prepared sequence, as a DataFrame (T rows)."""
    kp, scale = prep["kp"], prep["scale"]
    hip = midpoint(kp, L_HIP, R_HIP)
    sh = midpoint(kp, L_SH, R_SH)
    dx, dy = sh[:, 0] - hip[:, 0], sh[:, 1] - hip[:, 1]
    torso_angle = np.degrees(np.arctan2(np.abs(dx), -dy))
    hip_y = pd.Series(hip[:, 1] / scale)
    hip_vy = (hip_y.diff() * fps).rolling(3, min_periods=1).mean()
    hip_ay = hip_vy.diff() * fps
    b = prep["bbox"]
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    bbox_ratio = np.where(w > 0, h / np.where(w > 0, w, 1), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        head_y = np.nanmean(kp[:, 0:5, 1], axis=1)
    head_height = (hip[:, 1] - head_y) / scale
    return pd.DataFrame({"torso_angle": torso_angle, "hip_vy": hip_vy.to_numpy(),
                         "hip_ay": hip_ay.to_numpy(), "bbox_ratio": bbox_ratio,
                         "head_height": head_height})


def _first_last(w):
    """First and last non-NaN value of each row of w (n, L); NaN if the row is empty."""
    ok = ~np.isnan(w)
    any_ok = ok.any(axis=1)
    first = np.argmax(ok, axis=1)
    last = w.shape[1] - 1 - np.argmax(ok[:, ::-1], axis=1)
    rows = np.arange(len(w))
    f, l = w[rows, first], w[rows, last]
    f[~any_ok], l[~any_ok] = np.nan, np.nan
    return f, l


def window_features(prep, starts, length=CFG["window"]["length"]):
    """Features of the windows of one prepared sequence starting at the given frame indices."""
    starts = np.asarray(starts, dtype=int)
    sig = frame_signals(prep)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for s in SIGNALS:
            w = sliding_window_view(sig[s].to_numpy(dtype=float), length)[starts]
            first, last = _first_last(w)
            out[f"{s}_mean"] = np.nanmean(w, axis=1)
            out[f"{s}_min"] = np.nanmin(w, axis=1)
            out[f"{s}_max"] = np.nanmax(w, axis=1)
            out[f"{s}_std"] = np.nanstd(w, axis=1)
            out[f"{s}_delta"] = last - first
    det = sliding_window_view(prep["detected"].astype(float), length)[starts]
    out["valid_frac"] = det.mean(axis=1)
    return pd.DataFrame(out)[FEATURES]


def build(processed=PROCESSED, poses_dir=None):
    """Compute features for every window of processed/windows_meta.parquet."""
    meta = pd.read_parquet(Path(processed) / "windows_meta.parquet")
    parts = []
    for seq, m in meta.groupby("seq", sort=False):
        df = load_poses(seq) if poses_dir is None else load_poses(seq, poses_dir)
        prep = prepare_sequence(df)
        idx = pd.Index(prep["frames"])
        starts = idx.get_indexer(m["start_frame"])
        f = window_features(prep, starts)
        f.insert(0, "window_id", m["window_id"].to_numpy())
        parts.append(f)
    feats = pd.concat(parts, ignore_index=True)
    feats = feats.set_index("window_id").loc[meta["window_id"]].reset_index()
    feats.to_parquet(Path(processed) / "features.parquet", index=False)
    return feats


if __name__ == "__main__":
    f = build()
    print(f"{len(f)} windows, {len(FEATURES)} features")
    print(f.drop(columns="window_id").describe().T[["mean", "min", "max", "count"]].round(2))
