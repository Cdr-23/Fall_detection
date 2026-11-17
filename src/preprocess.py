"""Turn cached poses (poses/<seq>.parquet) into fixed-length windows shared by every model.

Pipeline, sequence by sequence:
  1. A keypoint is missing when its confidence is < min_conf (0.3) or when YOLO found nobody
     (person == 0). Missing keypoints and boxes become NaN.
  2. Gaps of at most max_gap frames (15 = 0.5 s), bounded on both sides, are filled by linear
     interpolation in time. Longer gaps (frequent when the person lies on the floor) and gaps at
     the start or end of a sequence stay empty: we do not invent poses.
  3. Normalisation: centre = mid-hips (kp11, kp12) of the frame; scale = median torso length of
     the sequence (mid-shoulders kp5/kp6 to mid-hips, over the frames with a real detection).
     Simplification: the scale uses the whole sequence, so this step is not strictly real time.
  4. Windows of `length` (30) frames with stride `stride` (5), never crossing two sequences.

Label rule (team contract, CLAUDE.md section 7.0). URFD frame labels: -1 not lying,
0 temporary pose, 1 lying. Classes:
  1 = fall      : label 0 in a fall-* sequence (exactly 30 frames per fall)
  2 = lying     : label 1, in any sequence
  0 = not fall  : label -1, and label 0 in an adl-* sequence (sitting down, bending: hard cases)
  frames without a label (NaN) are ignored.
URFD itself does not use label 0 for classification: this mapping is a team choice.
A window takes the class of its LAST frame (what is happening now, usable in real time);
windows whose last frame has no label are dropped.

Outputs (processed/, rows in the same order everywhere):
  windows_meta.parquet  window_id, seq, start_frame, end_frame, y, fold, kind, valid_frac
  X_seq.npy             float32 (N, 30, 17, 2) normalised (x, y); empty values set to 0
  valid.npy             bool (N, 30), True when the frame had a real detection (person == 1,
                        not interpolated)
  folds.csv             seq, fold  (StratifiedGroupKFold, 5 folds, group = sequence,
                        stratified on kind, shuffle=True, random_state=42)
  sequences.csv         seq, kind, fold, n_frames, fall_start, fall_end (frame numbers of the
                        fall phase, used by the event metrics)

Usage: python src/preprocess.py               # all sequences in poses/
       python src/preprocess.py fall-01 ...   # subset (folds are still computed on all 70)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "params.yaml").read_text())
POSES, PROCESSED = ROOT / "poses", ROOT / "processed"
N_KP = 17
L_HIP, R_HIP, L_SH, R_SH = 11, 12, 5, 6
CLASSES = {0: "not fall", 1: "fall", 2: "lying"}


def seq_kind(seq):
    """'fall' or 'adl', from the sequence name (fall-01, adl-12, gmdcsa-s1-fall-03, ...)."""
    return "fall" if "fall" in str(seq).split("-") else "adl"


def frame_classes(df):
    """Per-frame class (0/1/2, NaN when unlabelled). Uses column `y` if present (already a class,
    e.g. external datasets), otherwise maps the URFD `label` column with the team rule."""
    if "y" in df:
        return df["y"].to_numpy(dtype=float)
    label = df["label"].to_numpy(dtype=float)
    fall = df["seq"].map(seq_kind).to_numpy() == "fall"
    y = np.full(len(df), np.nan)
    y[label == -1] = 0
    y[(label == 0) & ~fall] = 0
    y[(label == 0) & fall] = 1
    y[label == 1] = 2
    return y


def interpolate_gaps(a, max_gap):
    """Linear interpolation along axis 0 of NaN runs of length <= max_gap that have a valid value
    on both sides. Works column by column on any (T, ...) array; returns a new array."""
    a = np.asarray(a, dtype=float)
    shape = a.shape
    flat = a.reshape(shape[0], -1).copy()
    t = np.arange(shape[0])
    for c in range(flat.shape[1]):
        x = flat[:, c]
        nan = np.isnan(x)
        if not nan.any() or nan.all():
            continue
        # run boundaries of NaN segments
        d = np.diff(np.r_[0, nan.astype(int), 0])
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]   # [start, end)
        ok = ~nan
        for s, e in zip(starts, ends):
            if s == 0 or e == len(x) or e - s > max_gap:
                continue
            x[s:e] = np.interp(t[s:e], t[ok], x[ok])
    return flat.reshape(shape)


def pose_arrays(df, min_conf):
    """Raw arrays from a pose DataFrame: keypoints (T,17,2) and bbox (T,4) with NaN when
    missing, and detected (T,) bool (person == 1)."""
    detected = df["person"].to_numpy() == 1
    xs = df[[f"kp{j}_x" for j in range(N_KP)]].to_numpy(dtype=float)
    ys = df[[f"kp{j}_y" for j in range(N_KP)]].to_numpy(dtype=float)
    cs = df[[f"kp{j}_c" for j in range(N_KP)]].to_numpy(dtype=float)
    miss = ~(cs >= min_conf) | ~detected[:, None]          # NaN confidence counts as missing
    kp = np.stack([xs, ys], axis=-1)
    kp[miss] = np.nan
    bbox = df[["bx1", "by1", "bx2", "by2"]].to_numpy(dtype=float, copy=True)
    bbox[~detected] = np.nan
    return kp, bbox, detected


def midpoint(kp, a, b):
    """Mid-point of keypoints a and b (T,2); falls back to the one available."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(kp[:, [a, b], :], axis=1)


def torso_scale(kp_raw, bbox_raw):
    """Median torso length (px) over frames where both shoulders and both hips are detected.
    Fallback when never visible: 0.3 x median box height."""
    hips = kp_raw[:, [L_HIP, R_HIP]].mean(axis=1)          # NaN unless both present
    sh = kp_raw[:, [L_SH, R_SH]].mean(axis=1)
    length = np.linalg.norm(sh - hips, axis=1)
    length = length[np.isfinite(length) & (length > 0)]
    if len(length):
        return float(np.median(length))
    h = bbox_raw[:, 3] - bbox_raw[:, 1]
    h = h[np.isfinite(h)]
    return float(0.3 * np.median(h)) if len(h) else np.nan


def prepare_sequence(df, cfg=CFG):
    """Clean one sequence. Returns a dict of per-frame arrays:
    frames (T,), y (T,) class or NaN, detected (T,), kp (T,17,2) interpolated pixels,
    bbox (T,4) interpolated, center (T,2), scale (float), kp_norm (T,17,2) normalised (NaN kept)."""
    df = df.sort_values("frame").reset_index(drop=True)
    p = cfg["preprocess"]
    kp_raw, bbox_raw, detected = pose_arrays(df, p["min_conf"])
    kp = interpolate_gaps(kp_raw, p["max_gap"])
    bbox = interpolate_gaps(bbox_raw, p["max_gap"])
    center = midpoint(kp, L_HIP, R_HIP)
    scale = torso_scale(kp_raw, bbox_raw)
    kp_norm = (kp - center[:, None, :]) / scale
    return {"seq": str(df["seq"].iloc[0]), "frames": df["frame"].to_numpy(), "y": frame_classes(df),
            "detected": detected, "kp": kp, "bbox": bbox, "center": center, "scale": scale,
            "kp_norm": kp_norm, "img_w": df["img_w"].iloc[0], "img_h": df["img_h"].iloc[0]}


def window_starts(n_frames, length, stride):
    """Start indices of the windows of one sequence."""
    return np.arange(0, n_frames - length + 1, stride)


def sequence_windows(prep, cfg=CFG, keep_unlabelled=False):
    """Windows of one prepared sequence: (meta rows, X (n,L,17,2) float32, valid (n,L) bool).
    Unlabelled windows (last frame without class) are dropped unless keep_unlabelled."""
    L, S = cfg["window"]["length"], cfg["window"]["stride"]
    rows, xs, vs = [], [], []
    X_all = np.nan_to_num(prep["kp_norm"], nan=0.0).astype(np.float32)
    for s in window_starts(len(prep["frames"]), L, S):
        e = s + L - 1
        y = prep["y"][e]
        if np.isnan(y) and not keep_unlabelled:
            continue
        v = prep["detected"][s:e + 1]
        rows.append({"seq": prep["seq"], "start_frame": int(prep["frames"][s]),
                     "end_frame": int(prep["frames"][e]), "y": y,
                     "kind": seq_kind(prep["seq"]), "valid_frac": float(v.mean())})
        xs.append(X_all[s:e + 1])
        vs.append(v)
    X = np.stack(xs) if xs else np.zeros((0, L, N_KP, 2), np.float32)
    V = np.stack(vs) if vs else np.zeros((0, L), bool)
    return rows, X, V


def make_folds(seqs, cfg=CFG):
    """Sequence-level folds: StratifiedGroupKFold, group = sequence, stratified on kind."""
    seqs = sorted(seqs)
    kinds = [seq_kind(s) for s in seqs]
    sgkf = StratifiedGroupKFold(n_splits=cfg["cv"]["n_folds"], shuffle=True,
                                random_state=cfg["seed"])
    fold = {}
    for k, (_, test) in enumerate(sgkf.split(np.zeros(len(seqs)), kinds, groups=seqs)):
        for i in test:
            fold[seqs[i]] = k
    return pd.DataFrame({"seq": seqs, "fold": [fold[s] for s in seqs]})


def load_poses(seq, poses_dir=POSES):
    return pd.read_parquet(Path(poses_dir) / f"{seq}.parquet")


def build(seqs=None, poses_dir=POSES, out=PROCESSED, cfg=CFG):
    """Write the processed/ files. Folds are always computed on every sequence of poses_dir."""
    all_seqs = sorted(p.stem for p in Path(poses_dir).glob("*.parquet"))
    folds = make_folds(all_seqs, cfg)
    fold_of = dict(zip(folds.seq, folds.fold))
    seqs = all_seqs if not seqs else sorted(seqs)
    metas, Xs, Vs, seq_rows = [], [], [], []
    for seq in seqs:
        prep = prepare_sequence(load_poses(seq, poses_dir), cfg)
        rows, X, V = sequence_windows(prep, cfg)
        metas += rows
        Xs.append(X)
        Vs.append(V)
        fall = prep["frames"][prep["y"] == 1]
        seq_rows.append({"seq": seq, "kind": seq_kind(seq), "fold": fold_of[seq],
                         "n_frames": len(prep["frames"]),
                         "fall_start": fall.min() if len(fall) else np.nan,
                         "fall_end": fall.max() if len(fall) else np.nan})
    meta = pd.DataFrame(metas)
    meta.insert(0, "window_id", np.arange(len(meta), dtype=np.int64))
    meta["y"] = meta["y"].astype(np.int64)
    meta["fold"] = meta["seq"].map(fold_of).astype(np.int64)
    meta = meta[["window_id", "seq", "start_frame", "end_frame", "y", "fold", "kind", "valid_frac"]]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(out / "windows_meta.parquet", index=False)
    np.save(out / "X_seq.npy", np.concatenate(Xs).astype(np.float32))
    np.save(out / "valid.npy", np.concatenate(Vs).astype(bool))
    folds.to_csv(out / "folds.csv", index=False)
    seq_df = pd.DataFrame(seq_rows)
    seq_df["fall_start"] = seq_df["fall_start"].astype("Int64")
    seq_df["fall_end"] = seq_df["fall_end"].astype("Int64")
    seq_df.to_csv(out / "sequences.csv", index=False)
    return meta


def summary(meta):
    """Windows per class and per fold (printed as a check)."""
    t = pd.crosstab(meta["fold"], meta["y"].map(CLASSES), margins=True)
    s = meta.groupby("fold")["seq"].nunique().rename("sequences")
    return t.join(s)


if __name__ == "__main__":
    m = build(sys.argv[1:] or None)
    print(f"{len(m)} windows from {m.seq.nunique()} sequences")
    print(summary(m))
