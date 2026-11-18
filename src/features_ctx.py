"""Context features (P7): height above the floor and a 3-second look-back, on top of the 26
base features of features.py. Experiment: aims at separating a fall from lying down on purpose.

Same conventions as features.py: image y points down, scale = median torso length of the
sequence (prep["scale"], whole sequence: not strictly real time, as in the base pipeline),
fps = 30, a window = frames [start, start + 29]. Every definition below only uses frames <= the
window's last frame (causal) on the prepared arrays. NaN stays NaN: models impute with the
TRAIN median. Definitions are fixed in advance (not tuned on any score).

A. FLOOR (height above the floor; 6 features)
  standing frame  : person detected (prep["detected"]), torso_angle < 30 deg, and both ankles
                    (kp 15 and 16) available in prep["kp"] (not NaN).
  floor_ref_y(t)  : median of the y (pixels) of the mid-ankles over the last 30 standing frames
                    of index <= t, anywhere earlier in the sequence; NaN while fewer than 10
                    standing frames have been seen.
  hip_above_floor(t)  = (floor_ref_y(t) - y_mid_hips(t)) / scale
  head_above_floor(t) = (floor_ref_y(t) - y_head(t)) / scale, head = mean of the available kp 0..4
  per window and per signal: _mean, _min, _last (last non-NaN value).

B. LOOKBACK, 3 s (6 features), on the 90 frames [end - 89, end] (truncated at the sequence start)
  hip_drop_3s        (y_hips at the last valid frame - min y_hips over the 3 s) / scale
  descent_time_3s    (end_frame - frame of the highest hips, i.e. min y, first one on ties) / fps
  hip_vy_max_3s, hip_ay_max_3s      peaks of the vertical speed / acceleration (frame_signals)
  torso_angle_min_3s, head_height_max_3s   was the person upright during the 3 s?

C. LOOKBACK, previous windows (8 features): for f in torso_angle_mean, head_height_mean,
  bbox_ratio_mean, hip_vy_max, the same statistic on the 30-frame window ending 30 frames
  earlier (f_lag1s) and 60 frames earlier (f_lag2s), computed from the per-frame signals; NaN
  when that window starts before the sequence.

Outputs: <processed_dir>/features_ctx.parquet (window_id + FEATURES_CTX), same row order as
windows_meta.parquet; the 26 base columns are checked equal to features.parquet.
Usage: python src/features_ctx.py        # processed/, processed/gmdcsa24, processed/le2i
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import features
from preprocess import CFG, L_HIP, POSES, PROCESSED, R_HIP, ROOT, load_poses, midpoint, \
    prepare_sequence

FPS, LENGTH = CFG["data"]["fps"], CFG["window"]["length"]
L_ANK, R_ANK = 15, 16
STANDING_ANGLE = 30.0          # degrees
FLOOR_LAST, FLOOR_MIN = 30, 10  # last standing frames used / minimum needed
LOOKBACK = 90                  # frames (3 s)
LAGS = {"lag1s": 30, "lag2s": 60}
LAGGED = ["torso_angle_mean", "head_height_mean", "bbox_ratio_mean", "hip_vy_max"]

FLOOR = [f"{s}_above_floor_{a}" for s in ("hip", "head") for a in ("mean", "min", "last")]
LOOK_3S = ["hip_drop_3s", "descent_time_3s", "hip_vy_max_3s", "hip_ay_max_3s",
           "torso_angle_min_3s", "head_height_max_3s"]
LOOK_LAG = [f"{f}_{lag}" for lag in LAGS for f in LAGGED]
LOOKBACK_FEATS = LOOK_3S + LOOK_LAG
CTX = FLOOR + LOOKBACK_FEATS
FEATURES_CTX = features.FEATURES + CTX
GROUPS = {"FLOOR": FLOOR, "LOOKBACK": LOOKBACK_FEATS}
SETS = {"urfd": (PROCESSED, POSES), "gmdcsa24": (PROCESSED / "gmdcsa24", POSES / "gmdcsa24"),
        "le2i": (PROCESSED / "le2i", POSES / "le2i")}


def _nan_reduce(fn, a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(a)


def ctx_signals(prep, sig=None):
    """Per-frame arrays used by the context features (T rows)."""
    kp, scale = prep["kp"], prep["scale"]
    sig = features.frame_signals(prep) if sig is None else sig
    hip_y = midpoint(kp, L_HIP, R_HIP)[:, 1]
    head_y = _nan_reduce(lambda a: np.nanmean(a, axis=1), kp[:, 0:5, 1])
    ankle_y = kp[:, [L_ANK, R_ANK], 1].mean(axis=1)          # NaN unless both ankles available
    angle = sig["torso_angle"].to_numpy(dtype=float)
    standing = prep["detected"] & (angle < STANDING_ANGLE) & np.isfinite(ankle_y)
    floor = np.full(len(hip_y), np.nan)
    st_idx = np.flatnonzero(standing)
    n_seen = np.cumsum(standing)                            # standing frames of index <= t
    for t in np.flatnonzero(n_seen >= FLOOR_MIN):
        k = n_seen[t]
        floor[t] = np.median(ankle_y[st_idx[max(0, k - FLOOR_LAST):k]])
    return {"hip_y": hip_y, "standing": standing, "floor_ref_y": floor,
            "hip_above_floor": (floor - hip_y) / scale,
            "head_above_floor": (floor - head_y) / scale, "sig": sig}


def _last_valid(a):
    ok = np.flatnonzero(np.isfinite(a))
    return a[ok[-1]] if len(ok) else np.nan


def context_features(prep, starts, length=LENGTH, sig=None):
    """The 20 context features (DataFrame, CTX columns) of the windows starting at the given
    frame indices of one prepared sequence. Only frames <= each window's end are read."""
    starts = np.asarray(starts, dtype=int)
    c = ctx_signals(prep, sig)
    sig, scale, frames = c["sig"], prep["scale"], prep["frames"]
    s_arr = {k: sig[k].to_numpy(dtype=float) for k in features.SIGNALS}
    rows = []
    for s in starts:
        e = s + length - 1
        r = {}
        for name in ("hip", "head"):
            w = c[f"{name}_above_floor"][s:e + 1]
            r[f"{name}_above_floor_mean"] = _nan_reduce(np.nanmean, w)
            r[f"{name}_above_floor_min"] = _nan_reduce(np.nanmin, w)
            r[f"{name}_above_floor_last"] = _last_valid(w)
        b = max(0, e - LOOKBACK + 1)
        hy = c["hip_y"][b:e + 1]
        if np.isfinite(hy).any():
            top = int(np.nanargmin(hy))                     # first frame of the highest hips
            r["hip_drop_3s"] = (_last_valid(hy) - hy[top]) / scale
            r["descent_time_3s"] = (frames[e] - frames[b + top]) / FPS
        else:
            r["hip_drop_3s"] = r["descent_time_3s"] = np.nan
        r["hip_vy_max_3s"] = _nan_reduce(np.nanmax, s_arr["hip_vy"][b:e + 1])
        r["hip_ay_max_3s"] = _nan_reduce(np.nanmax, s_arr["hip_ay"][b:e + 1])
        r["torso_angle_min_3s"] = _nan_reduce(np.nanmin, s_arr["torso_angle"][b:e + 1])
        r["head_height_max_3s"] = _nan_reduce(np.nanmax, s_arr["head_height"][b:e + 1])
        for lag, d in LAGS.items():
            le, ls = e - d, e - d - length + 1
            for f in LAGGED:
                if ls < 0:
                    r[f"{f}_{lag}"] = np.nan
                    continue
                signal, agg = f.rsplit("_", 1)
                fn = np.nanmean if agg == "mean" else np.nanmax
                r[f"{f}_{lag}"] = _nan_reduce(fn, s_arr[signal][ls:le + 1])
        rows.append(r)
    return pd.DataFrame(rows, columns=CTX, dtype=float)


def window_features_ctx(prep, starts, length=LENGTH):
    """The 46 columns FEATURES_CTX (26 base + 20 context) of the given windows."""
    sig = features.frame_signals(prep)
    base = features.window_features(prep, starts, length)
    return pd.concat([base, context_features(prep, starts, length, sig)], axis=1)[FEATURES_CTX]


def build(processed_dir=PROCESSED, poses_dir=POSES):
    """Write <processed_dir>/features_ctx.parquet, same row order as windows_meta.parquet."""
    processed_dir = Path(processed_dir)
    meta = pd.read_parquet(processed_dir / "windows_meta.parquet")
    parts = []
    for seq, m in meta.groupby("seq", sort=False):
        prep = prepare_sequence(load_poses(seq, poses_dir))
        starts = pd.Index(prep["frames"]).get_indexer(m["start_frame"])
        assert (starts >= 0).all(), seq
        f = window_features_ctx(prep, starts)
        f.insert(0, "window_id", m["window_id"].to_numpy())
        parts.append(f)
    feats = pd.concat(parts, ignore_index=True)
    feats = feats.set_index("window_id").loc[meta["window_id"]].reset_index()
    base = pd.read_parquet(processed_dir / "features.parquet")
    assert (base["window_id"].to_numpy() == feats["window_id"].to_numpy()).all()
    a, b = base[features.FEATURES].to_numpy(float), feats[features.FEATURES].to_numpy(float)
    if not np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True):
        raise AssertionError(f"{processed_dir}: base features differ from features.parquet "
                             f"(max abs diff {np.nanmax(np.abs(a - b)):.3g})")
    # recomputation matches up to float rounding (~1e-13): keep the stored base columns so
    # that they are bit-identical to what the retained models were trained on
    feats[features.FEATURES] = base[features.FEATURES].to_numpy()
    feats.to_parquet(processed_dir / "features_ctx.parquet", index=False)
    return feats


if __name__ == "__main__":
    for name, (proc, poses) in SETS.items():
        f = build(proc, poses)
        print(f"{name}: {len(f)} windows, {len(FEATURES_CTX)} features "
              f"-> {Path(proc).relative_to(ROOT) / 'features_ctx.parquet'}")
        print(f[CTX].describe().T[["mean", "min", "max", "count"]].round(2).to_string())
