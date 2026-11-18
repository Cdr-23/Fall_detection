"""Run YOLOv8n-pose on videos and cache the keypoints.

Default: the RGB half of every URFD video. Output: poses/<seq>.parquet, one row per frame,
with columns
  seq, frame (1-based, same numbering as the label CSVs), label, person (0/1),
  kp0_x, kp0_y, kp0_c ... kp16_x, kp16_y, kp16_c   (COCO order, pixels, confidence),
  bx1, by1, bx2, by2 (bbox of the most confident person), img_w, img_h.
label: -1 not lying, 0 falling, 1 lying (NaN when the frame is absent from the label CSV).
Frames where YOLO finds nobody have person=0 and NaN keypoints.

Any other video (GMDCSA-24, the team's videos): extract_video() uses the FULL image and, when
the video is not at 30 fps, resamples the poses to 30 fps by linear interpolation between the
two nearest frames, so that a 30-frame window always lasts 1 s (see src/external.py).

Usage: python src/extract_pose.py [seq ...]           URFD (default: all, skips cached ones)
       python src/extract_pose.py --video v.mp4 out.parquet   any video, full image
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "params.yaml").read_text())
RAW, OUT = ROOT / "data" / "raw", ROOT / "poses"
KP_COLS = [f"kp{j}_{a}" for j in range(17) for a in "xyc"]
BOX_COLS = ["bx1", "by1", "bx2", "by2"]


def device():
    d = CFG["pose"]["device"]
    if d != "auto":
        return d
    if torch.cuda.is_available():
        return "cuda"                                   # Windows / Linux with an NVIDIA GPU
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model():
    return YOLO(str(ROOT / CFG["pose"]["model"]))      # weights cached at the repo root


def pose_row(model, img, dev):
    """Keypoints of the most confident person in one image, as a dict."""
    h, w = img.shape[:2]
    r = model.predict(img, device=dev, verbose=False, conf=CFG["pose"]["conf"])[0]
    row = {"img_w": w, "img_h": h, "person": 0}
    if len(r.boxes):
        k = int(r.boxes.conf.argmax())                 # keep the most confident person
        kp, b = r.keypoints.data[k].cpu().numpy(), r.boxes.xyxy[k].cpu().numpy()
        row.update(person=1, bx1=b[0], by1=b[1], bx2=b[2], by2=b[3])
        for j in range(17):
            row[f"kp{j}_x"], row[f"kp{j}_y"], row[f"kp{j}_c"] = kp[j]
    return row


def load_labels():
    cols = ["seq", "frame", "label"]
    return pd.concat([pd.read_csv(RAW / f, header=None, usecols=[0, 1, 2], names=cols)
                      for f in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv")])


def process(video, model, labels, dev):
    """One URFD video: right half (RGB) only, frames kept as they are."""
    seq = video.name.replace("-cam0.mp4", "")
    out = OUT / f"{seq}.parquet"
    if out.exists():
        return f"{seq}: cached"
    rows, cap, i = [], cv2.VideoCapture(str(video)), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        rgb = frame[:, frame.shape[1] // 2:]           # right half = RGB, left half = depth
        rows.append({"seq": seq, "frame": i, **pose_row(model, rgb, dev)})
    df = pd.DataFrame(rows).merge(labels[labels.seq == seq][["frame", "label"]], on="frame", how="left")
    df.to_parquet(out)
    return f"{seq}: {len(df)} frames, person found in {df.person.mean():.0%}"


def resample(df, src_fps, fps):
    """Resample a per-frame pose table from src_fps to fps by linear interpolation between the
    two nearest source frames (a keypoint missing in either stays missing; confidence = min)."""
    n = len(df)
    t = np.arange(int(np.floor((n - 1) / src_fps * fps)) + 1) / fps
    u = t * src_fps
    i0 = np.clip(np.floor(u).astype(int), 0, n - 1)
    i1 = np.clip(i0 + 1, 0, n - 1)
    w = (u - i0)[:, None]
    exact = np.isclose(w[:, 0], 0)
    out = pd.DataFrame({"frame": np.arange(1, len(t) + 1)})
    for cols, how in ((BOX_COLS, "lin"), ([c for c in KP_COLS if c.endswith(("_x", "_y"))], "lin"),
                      ([c for c in KP_COLS if c.endswith("_c")], "min")):
        a0, a1 = df[cols].to_numpy(float)[i0], df[cols].to_numpy(float)[i1]
        v = (1 - w) * a0 + w * a1 if how == "lin" else np.minimum(a0, a1)
        v[exact] = a0[exact]
        out[cols] = v
    p0, p1 = df["person"].to_numpy()[i0], df["person"].to_numpy()[i1]
    out["person"] = np.where(exact, p0, p0 * p1)
    out["img_w"], out["img_h"] = df["img_w"].iloc[0], df["img_h"].iloc[0]
    return out


def extract_video(path, model=None, dev=None, seq=None, crop="full", fps=None):
    """Poses of any video (full image, or 'right' half like URFD), resampled to `fps`
    (default 30) when the video has another frame rate. Returns the pose DataFrame."""
    model, dev = model or load_model(), dev or device()
    fps = fps or CFG["data"]["fps"]
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    rows = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if crop == "right":
            frame = frame[:, frame.shape[1] // 2:]
        rows.append(pose_row(model, frame, dev))
    df = pd.DataFrame(rows)
    for c in KP_COLS + BOX_COLS:
        if c not in df:
            df[c] = np.nan
    if abs(src_fps - fps) > 0.5:
        df = resample(df, src_fps, fps)
    else:
        df.insert(0, "frame", np.arange(1, len(df) + 1))
    df.insert(0, "seq", seq or Path(path).stem)
    df["src_fps"] = src_fps
    cols = ["seq", "frame", "person"] + KP_COLS + BOX_COLS + ["img_w", "img_h", "src_fps"]
    return df[cols]


def main():
    if sys.argv[1:2] == ["--video"]:
        video, out = sys.argv[2], Path(sys.argv[3])
        out.parent.mkdir(parents=True, exist_ok=True)
        df = extract_video(video)
        df.to_parquet(out)
        print(f"{out}: {len(df)} frames, person found in {df.person.mean():.0%}")
        return
    OUT.mkdir(exist_ok=True)
    labels, dev = load_labels(), device()
    model = load_model()
    only = set(sys.argv[1:])
    for v in sorted(RAW.glob("*-cam0.mp4")):
        if only and v.name.replace("-cam0.mp4", "") not in only:
            continue
        print(process(v, model, labels, dev), flush=True)


if __name__ == "__main__":
    main()
