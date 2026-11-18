"""External test on GMDCSA-24 (CLAUDE.md §5, §8).

Models are fitted on URFD only (all 70 sequences). Rule thresholds and the
logistic-regression C are chosen on URFD exactly as in src/p2.py. GMDCSA-24
is only used to score the frozen models. Nothing here is tuned on it.

GMDCSA-24 label format: per-clip free-text `Classes` column with intervals,
e.g. "Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]". Frame classes:
  - inside a `Falling` interval          -> chute (1)
  - in a Fall clip, after the fall ends  -> allonge (2)
  - everything else with a label         -> pas chute (0)
  - frames not covered by any interval   -> skipped
Sleeping in bed is pas chute: CLAUDE.md defines allonge as lying on the
ground after a fall.

Run from the repo root:  python src/external.py [--no-extract]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

import p2
from extract_poses import extract_one
from metrics import CHUTE, event_metrics, window_metrics
from prepare import CFG, ROOT, frame_signals, load_windows, window_features

GMD = ROOT / "data" / "raw" / "gmdcsa24" / "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76"
POSE_DIR = ROOT / "data" / "pose_gmdcsa"
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "data" / "yolov8n-pose.pt"
ALLONGE = 2

SEG = re.compile(r"([A-Za-z][A-Za-z ()/]*?)\s*\[\s*([\d.]+)\s+to\s+([\d.]+)\s*\]")


def seq_name(subj: str, kind: str, fname: str) -> str:
    return f"S{subj}-{kind}-{fname[:-4]}"


def parse_clip_table() -> dict:
    """{(subject, kind, filename): [(label, start_s, end_s), ...]}"""
    out = {}
    for subj_dir in sorted(GMD.glob("Subject *")):
        subj = subj_dir.name.split()[-1]
        for kind in ["ADL", "Fall"]:
            csv = subj_dir / f"{kind}.csv"
            for line in csv.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
                m = re.match(r"\s*([\w\-]+\.mp4)", line)
                if not m:
                    continue
                segs = [(n.strip(), float(s), float(e)) for n, s, e in SEG.findall(line)]
                out[(subj, kind, m.group(1))] = segs
    return out


def frame_classes(segs: list, is_fall_clip: bool, n_frames: int, fps: float) -> np.ndarray:
    t = np.arange(n_frames) / fps
    cls = np.full(n_frames, -1)
    fall_end = None
    for label, s, e in segs:
        m = (t >= s) & (t <= e)
        if "fall" in label.lower():
            cls[m] = CHUTE
            fall_end = e if fall_end is None else max(fall_end, e)
        else:
            cls[m] = 0
    if is_fall_clip and fall_end is not None:
        cls[t > fall_end] = ALLONGE
    return cls


def extract_all(device: str) -> None:
    POSE_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    for (subj, kind, fname) in parse_clip_table():
        out = POSE_DIR / f"{seq_name(subj, kind, fname)}.npz"
        if out.exists():
            continue
        video = GMD / f"Subject {subj}" / kind / fname
        extract_one(model, video, out, device, batch=16, crop_depth=False)
        print("pose:", out.name)


def build_external_windows() -> tuple[pd.DataFrame, pd.DataFrame]:
    win, step = int(CFG["window"]), int(CFG["step"])
    rows, clips = [], []
    for (subj, kind, fname), segs in parse_clip_table().items():
        seq = seq_name(subj, kind, fname)
        npz = POSE_DIR / f"{seq}.npz"
        meta = {"seq": seq, "subject": subj, "kind": kind, "n_segments": len(segs)}
        if not npz.exists() or not segs:
            clips.append({**meta, "status": "no pose" if not npz.exists() else "no labels"})
            continue
        if kind == "Fall" and not any("fall" in n.lower() for n, _, _ in segs):
            clips.append({**meta, "status": "fall clip without fall label (excluded)"})
            continue
        z = np.load(npz)
        fps = float(z["fps"])
        sig = frame_signals(z["kp"], z["box"], fps)
        if sig is None:
            clips.append({**meta, "status": "no usable person"})
            continue
        cls = frame_classes(segs, kind == "Fall", len(sig), fps)
        counts = {k: int((cls == k).sum()) for k in [0, CHUTE, ALLONGE]}
        n_windows = 0
        for start in range(0, len(sig) - win + 1, step):
            end = start + win - 1
            y = cls[end]
            if y < 0:
                continue
            rows.append({
                "seq": seq, "is_fall": kind == "Fall", "start": start, "end": end,
                "fps": fps, "n_frames": len(sig), "y": int(y),
                **window_features(sig[start : end + 1]),
            })
            n_windows += 1
        clips.append({**meta, "status": "ok", "fps": fps, "n_frames": len(sig),
                      "frames_pas_chute": counts[0], "frames_chute": counts[CHUTE],
                      "frames_allonge": counts[ALLONGE], "n_windows": n_windows})
    return pd.DataFrame(rows), pd.DataFrame(clips)


def evaluate(train: pd.DataFrame, test: pd.DataFrame) -> None:
    specs = {
        "p2-rules": (lambda: p2.fit_rules(train), p2.predict_rules),
        "p2-logreg": (lambda: p2.fit_logreg(train), p2.predict_logreg),
    }
    for name, (fit, predict) in specs.items():
        model, info = fit()
        t0 = time.perf_counter()
        pred, proba = predict(model, test)
        ms = (time.perf_counter() - t0) * 1000
        frames = test.drop_duplicates("seq")["n_frames"].sum()
        row = {
            "algo": name, "train": "URFD all 70 seq", "test": "GMDCSA-24",
            **window_metrics(test["y"].to_numpy(), proba, pred),
            **event_metrics(test, pred),
            "infer_ms_per_frame": ms / frames,
            "model_size_bytes": p2.model_size_bytes(name, model, info),
            "params": json.dumps(info),
        }
        res = pd.DataFrame([row])
        res.to_csv(RESULTS / f"{name}_gmdcsa.csv", index=False)
        pd.DataFrame({
            "seq": test["seq"].to_numpy(), "end": test["end"].to_numpy(),
            "y": test["y"].to_numpy(), "pred": pred,
            "p_pas_chute": proba[:, 0], "p_chute": proba[:, 1], "p_allonge": proba[:, 2],
        }).to_csv(RESULTS / f"{name}_gmdcsa_windows.csv", index=False)

        print(f"\n== {name} on GMDCSA-24 (URFD-fitted, frozen) ==")
        print(f"  params: {row['params']}")
        for k in ["roc_auc", "pr_auc", "recall_chute", "precision_chute", "f1_chute", "macro_f1",
                  "event_recall", "delay_s_mean", "n_false_alerts", "adl_seconds",
                  "false_alerts_per_h"]:
            print(f"  {k:20s} {row[k]:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-extract", action="store_true", help="skip pose extraction")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    if not args.no_extract:
        extract_all(args.device)

    test, clips = build_external_windows()
    RESULTS.mkdir(exist_ok=True)
    clips.to_csv(RESULTS / "gmdcsa_clips.csv", index=False)
    print("Clip status:", clips["status"].value_counts().to_dict())
    print("Windows:", len(test), "| label counts:",
          test["y"].map({0: "pas_chute", 1: "chute", 2: "allonge"}).value_counts().to_dict())
    train = load_windows()
    evaluate(train, test)


if __name__ == "__main__":
    main()
