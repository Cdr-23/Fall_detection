"""Extract 2D skeletons (17 COCO keypoints) from URFD videos with YOLOv8n-pose.

Each MP4 holds depth (left half) and RGB (right half). Only the RGB half is
fed to YOLO. For each frame we keep the largest detected person.

Output: data/pose/<video_stem>.npz with
    kp   float32 (T, 17, 3)  x, y, confidence  (NaN when no person found)
    box  float32 (T, 4)      x1, y1, x2, y2     (NaN when no person found)
    fps  float32 scalar
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def extract_one(model: YOLO, video: Path, out: Path, device: str, batch: int) -> None:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    kps, boxes = [], []
    buf: list[np.ndarray] = []

    def flush() -> None:
        if not buf:
            return
        results = model(buf, device=device, verbose=False)
        for r in results:
            if r.keypoints is None or len(r.boxes) == 0:
                kps.append(np.full((17, 3), np.nan, np.float32))
                boxes.append(np.full(4, np.nan, np.float32))
                continue
            areas = (r.boxes.xyxy[:, 2] - r.boxes.xyxy[:, 0]) * (r.boxes.xyxy[:, 3] - r.boxes.xyxy[:, 1])
            i = int(torch.argmax(areas))
            xy = r.keypoints.xy[i].cpu().numpy()
            conf = r.keypoints.conf[i].cpu().numpy()
            kps.append(np.concatenate([xy, conf[:, None]], axis=1).astype(np.float32))
            boxes.append(r.boxes.xyxy[i].cpu().numpy().astype(np.float32))
        buf.clear()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        rgb = frame[:, w // 2 :]  # right half = RGB
        buf.append(rgb)
        if len(buf) == batch:
            flush()
    flush()
    cap.release()

    np.savez_compressed(out, kp=np.stack(kps), box=np.stack(boxes), fps=np.float32(fps))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0", help="'0' for first CUDA GPU, 'cpu' otherwise")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--weights", default="yolov8n-pose.pt")
    args = ap.parse_args()

    raw = ROOT / "data" / "raw"
    out_dir = ROOT / "data" / "pose"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    videos = sorted(raw.glob("*-cam0.mp4"))
    if not videos:
        raise SystemExit(f"No URFD videos found in {raw}")

    for v in tqdm(videos, desc="pose"):
        out = out_dir / f"{v.stem}.npz"
        if out.exists():
            continue  # cached
        extract_one(model, v, out, args.device, args.batch)

    print(f"Done: {len(videos)} videos -> {out_dir}")


if __name__ == "__main__":
    main()
