"""Annotated images of typical errors for the slides (P5, CLAUDE.md 7.5 step 4).

For each case: 4 frames of the alerted (or missed) 30-frame window, RGB half of the URFD video
with the YOLO skeleton, the true class of each frame and the model's p1 at the end of the window.
Cases are picked from results/errors.csv for one model (default: boosting, the retained model):
the first false alerts followed or not by lying down, and the fall missed by most models.

Outputs: results/errors/<algo>_<seq>.png. Needs data/raw (python src/download.py).
Usage: python src/error_images.py [algo]
"""
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import ROOT  # noqa: E402

RAW, OUT = ROOT / "data" / "raw", ROOT / "results" / "errors"
BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
         (12, 14), (14, 16), (0, 5), (0, 6)]
NAMES = {-1: "not lying", 0: "temporary pose", 1: "lying"}


def read_frames(seq, frames):
    cap = cv2.VideoCapture(str(RAW / f"{seq}-cam0.mp4"))
    out = {}
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)          # pose frames are 1-based
        ok, img = cap.read()
        if ok:
            out[f] = cv2.cvtColor(img[:, img.shape[1] // 2:], cv2.COLOR_BGR2RGB)
    cap.release()
    return out


def draw(seq, end, title, path, labels):
    poses = pd.read_parquet(ROOT / "poses" / f"{seq}.parquet").set_index("frame")
    frames = [end - 29, end - 19, end - 9, end]
    imgs = read_frames(seq, frames)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, f in zip(axes, frames):
        ax.axis("off")
        if f not in imgs:
            continue
        ax.imshow(imgs[f])
        if f in poses.index and poses.loc[f, "person"] == 1:
            kp = np.array([[poses.loc[f, f"kp{i}_x"], poses.loc[f, f"kp{i}_y"]] for i in range(17)])
            for a, b in BONES:
                ax.plot(kp[[a, b], 0], kp[[a, b], 1], "-", color="lime", lw=1.5)
        else:
            ax.text(5, 20, "no person detected", color="red", fontsize=8)
        ax.set_title(f"frame {f} ({(f - end + 29) / 30:.1f} s): {NAMES.get(labels.get(f), '?')}",
                     fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(algo="boosting"):
    OUT.mkdir(parents=True, exist_ok=True)
    lab = pd.concat([pd.read_csv(RAW / f"urfall-cam0-{k}.csv", header=None).iloc[:, :3]
                     for k in ("adls", "falls")])
    lab.columns = ["seq", "frame", "label"]
    err = pd.read_csv(ROOT / "results" / "errors.csv")
    mine = err[err["algo"] == algo]
    lying_after = {}
    for r in mine[mine["error"] == "false alert"].itertuples():
        nxt = lab[(lab.seq == r.seq) & lab.frame.between(r.end_frame, r.end_frame + 60)]
        lying_after[r.seq] = bool((nxt.label == 1).any())
    fa = mine[mine["error"] == "false alert"]
    picks = [fa[fa.seq.map(lying_after)].head(1), fa[~fa.seq.map(lying_after)].head(2)]
    missed = err[err["error"] == "missed fall"]
    top_missed = missed["seq"].value_counts().index[0]
    picks.append(missed[missed.seq == top_missed].head(1).assign(algo=algo))
    oof = pd.read_parquet(ROOT / "results" / f"{algo}_oof.parquet")
    for r in pd.concat(picks).itertuples():
        labels = dict(lab[lab.seq == r.seq][["frame", "label"]].itertuples(index=False))
        win = oof[(oof.seq == r.seq) & (oof.end_frame == r.end_frame)]
        p1 = win["p1"].iloc[0] if len(win) else r.p1
        what = "FALSE ALERT" if r.error == "false alert" else "MISSED FALL"
        title = f"{what} - {r.seq}, model {algo}: p(fall) = {p1:.2f} at the end of the window"
        path = OUT / f"{algo}_{r.seq}.png"
        draw(r.seq, int(r.end_frame), title, path, labels)
        print("->", path)


if __name__ == "__main__":
    main(*sys.argv[1:])
