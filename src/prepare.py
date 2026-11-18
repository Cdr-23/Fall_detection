"""Prepare URFD windows and features from cached skeletons.

Pipeline (CLAUDE.md §6):
  1. Load data/pose/<seq>-cam0.npz and the URFD label CSVs.
  2. Drop low-confidence keypoints, interpolate missing values in time.
  3. Normalise: centre = mid-hips, scale = median torso length of the sequence.
  4. Per-frame signals: torso angle, vertical hip speed/acceleration,
     box height/width ratio, head height above hips.
  5. Windows of `window` frames, stride `step`. Window label = label of the
     last frame (proposal in CLAUDE.md §9).
  6. Window statistics (mean, std, min, max, last) and StratifiedGroupKFold
     folds with the sequence as group.

Output: data/windows.parquet (one row per window).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "params.yaml").read_text(encoding="utf-8"))

MIN_CONF = 0.3  # keypoints below this confidence are treated as missing
FRAME_SIGNALS = ["torso", "vy", "ay", "box_hw", "head_h"]
STATS = ["mean", "std", "min", "max", "last"]
FEATURE_COLS = [f"{s}_{st}" for s in FRAME_SIGNALS for st in STATS]

KP_NOSE, KP_L_SHOULDER, KP_R_SHOULDER, KP_L_HIP, KP_R_HIP = 0, 5, 6, 11, 12


def load_urfd_labels() -> pd.DataFrame:
    """Return one row per (seq, frame_idx) with the URFD label and class."""
    raw = ROOT / "data" / "raw"
    # Files have no header. Columns: seq, frame, label, then depth-derived
    # columns we do not use (8 in the actual files; only the first 3 are read).
    frames = []
    for name in ["urfall-cam0-falls.csv", "urfall-cam0-adls.csv"]:
        df = pd.read_csv(raw / name, header=None, usecols=[0, 1, 2],
                         names=["seq", "frame", "label"])
        frames.append(df)
    lab = pd.concat(frames, ignore_index=True)
    lab["frame_idx"] = lab["frame"] - 1  # CSV frames are 1-based
    lab["cls"] = lab["label"].map({int(k): v for k, v in CFG["urfd_label_to_class"].items()})
    if lab["cls"].isna().any():
        raise ValueError("Unknown URFD label value")
    return lab


def _interp(a: np.ndarray) -> np.ndarray:
    """Linear interpolation along time for each column; edges held constant."""
    return pd.DataFrame(a).interpolate(limit_direction="both").to_numpy()


def frame_signals(kp: np.ndarray, box: np.ndarray, fps: float) -> np.ndarray | None:
    """Return (T, len(FRAME_SIGNALS)) per-frame signals, or None if no person was seen."""
    xy = kp[:, :, :2].copy()
    xy[kp[:, :, 2] < MIN_CONF] = np.nan
    T = xy.shape[0]
    xy = _interp(xy.reshape(T, -1)).reshape(T, 17, 2)
    box = _interp(box)
    if np.isnan(xy).all() or np.isnan(box).all():
        return None

    hips = (xy[:, KP_L_HIP] + xy[:, KP_R_HIP]) / 2
    shoulders = (xy[:, KP_L_SHOULDER] + xy[:, KP_R_SHOULDER]) / 2
    torso_vec = shoulders - hips
    torso_len = np.linalg.norm(torso_vec, axis=1)
    scale = np.nanmedian(torso_len)
    if not np.isfinite(scale) or scale <= 0:
        return None

    # Angle from vertical: 0 = upright, 90 = horizontal.
    torso = np.degrees(np.arctan2(np.abs(torso_vec[:, 0]), np.abs(torso_vec[:, 1])))
    hip_y = hips[:, 1] / scale                          # torso units, y grows downward
    vy = np.gradient(hip_y) * fps                       # > 0 = moving down
    ay = np.gradient(vy) * fps
    box_hw = (box[:, 3] - box[:, 1]) / np.maximum(box[:, 2] - box[:, 0], 1e-6)
    head_h = (hips[:, 1] - xy[:, KP_NOSE, 1]) / scale  # > 0 = head above hips

    return np.column_stack([torso, vy, ay, box_hw, head_h]).astype(np.float64)


def window_features(sig: np.ndarray) -> dict[str, float]:
    out = {}
    for j, name in enumerate(FRAME_SIGNALS):
        s = sig[:, j]
        out[f"{name}_mean"] = s.mean()
        out[f"{name}_std"] = s.std()
        out[f"{name}_min"] = s.min()
        out[f"{name}_max"] = s.max()
        out[f"{name}_last"] = s[-1]
    return out


def build_windows() -> pd.DataFrame:
    fps_default = float(CFG["fps"])
    win, step = int(CFG["window"]), int(CFG["step"])
    labels = load_urfd_labels()
    pose_dir = ROOT / "data" / "pose"

    rows = []
    skipped = []
    for seq, g in labels.groupby("seq"):
        npz_path = pose_dir / f"{seq}-cam0.npz"
        if not npz_path.exists():
            skipped.append((seq, "no pose file"))
            continue
        z = np.load(npz_path)
        fps = float(z["fps"]) or fps_default
        sig = frame_signals(z["kp"], z["box"], fps)
        if sig is None:
            skipped.append((seq, "no usable person"))
            continue

        cls = np.full(len(sig), -1)
        idx = g["frame_idx"].to_numpy()
        keep = idx < len(sig)
        cls[idx[keep]] = g["cls"].to_numpy()[keep]
        if (~keep).any():
            skipped.append((seq, f"{(~keep).sum()} label rows beyond video length"))

        is_fall = seq.startswith("fall")
        for start in range(0, len(sig) - win + 1, step):
            end = start + win - 1
            y = cls[end]
            if y < 0:
                continue
            feats = window_features(sig[start : end + 1])
            rows.append({
                "seq": seq, "is_fall": is_fall, "start": start, "end": end,
                "fps": fps, "n_frames": len(sig), "y": int(y), **feats,
            })

    if skipped:
        print("Skipped:", skipped)
    df = pd.DataFrame(rows)
    df = assign_folds(df)
    return df


def assign_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Sequence-grouped, label-stratified folds (CLAUDE.md §8)."""
    sgk = StratifiedGroupKFold(
        n_splits=int(CFG["n_folds"]), shuffle=True, random_state=int(CFG["seed"])
    )
    df = df.copy()
    df["fold"] = -1
    for k, (_, te) in enumerate(sgk.split(df, df["y"], groups=df["seq"])):
        df.iloc[te, df.columns.get_loc("fold")] = k
    return df


def load_windows() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "windows.parquet")


if __name__ == "__main__":
    out = ROOT / "data" / "windows.parquet"
    df = build_windows()
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} windows to {out}")
    print(df.groupby("y").size().rename({0: "pas_chute", 1: "chute", 2: "allonge"}))
    print(df.groupby("fold").size())
