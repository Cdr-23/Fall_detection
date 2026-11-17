"""Run YOLOv8n-pose on the RGB half of every URFD video and cache the keypoints.

Output: poses/<seq>.parquet, one row per frame, with columns
  seq, frame (1-based, same numbering as the label CSVs), label, person (0/1),
  kp0_x, kp0_y, kp0_c ... kp16_x, kp16_y, kp16_c   (COCO order, pixels, confidence),
  bx1, by1, bx2, by2 (bbox of the most confident person), img_w, img_h.
label: -1 not lying, 0 falling, 1 lying (NaN when the frame is absent from the label CSV).
Frames where YOLO finds nobody have person=0 and NaN keypoints.

Usage: python src/extract_pose.py [seq ...]   (default: all sequences, skips cached ones)
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


def device():
    d = CFG["pose"]["device"]
    if d != "auto":
        return d
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_labels():
    cols = ["seq", "frame", "label"]
    return pd.concat([pd.read_csv(RAW / f, header=None, usecols=[0, 1, 2], names=cols)
                      for f in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv")])


def process(video, model, labels, dev):
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
        h, w = rgb.shape[:2]
        r = model.predict(rgb, device=dev, verbose=False, conf=CFG["pose"]["conf"])[0]
        row = {"seq": seq, "frame": i, "img_w": w, "img_h": h, "person": 0}
        if len(r.boxes):
            k = int(r.boxes.conf.argmax())             # keep the most confident person
            kp, b = r.keypoints.data[k].cpu().numpy(), r.boxes.xyxy[k].cpu().numpy()
            row.update(person=1, bx1=b[0], by1=b[1], bx2=b[2], by2=b[3])
            for j in range(17):
                row[f"kp{j}_x"], row[f"kp{j}_y"], row[f"kp{j}_c"] = kp[j]
        rows.append(row)
    df = pd.DataFrame(rows).merge(labels[labels.seq == seq][["frame", "label"]], on="frame", how="left")
    df.to_parquet(out)
    return f"{seq}: {len(df)} frames, person found in {df.person.mean():.0%}"


def main():
    OUT.mkdir(exist_ok=True)
    labels, dev = load_labels(), device()
    model = YOLO(CFG["pose"]["model"])
    only = set(sys.argv[1:])
    for v in sorted(RAW.glob("*-cam0.mp4")):
        if only and v.name.replace("-cam0.mp4", "") not in only:
            continue
        print(process(v, model, labels, dev), flush=True)


if __name__ == "__main__":
    main()
