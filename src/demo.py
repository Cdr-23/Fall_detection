"""Demo / ultimate test: one video -> poses -> windows -> features -> every algorithm.

For each algorithm that has both src/run_<algo>.py (with predict_proba) and models/<algo>.*,
prints and saves the predicted class over time (0 not fall / 1 fall / 2 lying) and p1.

Usage:
  python src/demo.py VIDEO.mp4 [--labels "Falling[3.4 to 4.4]; Lying[4.4 to 10]"]
                     [--algos naive,logreg,...] [--tau 0.5] [--crop full|right]
  python src/demo.py --poses poses/gmdcsa24/gmdcsa-s1-fall-01.parquet   # skip YOLO
URFD videos (*-cam0.mp4) are cropped to their RGB right half automatically.

Outputs: results/demo/<name>.csv (one row per window: time, class and p0/p1/p2 per algo)
and results/demo/<name>.png (timeline). Alerts fire when p1 >= tau.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import external  # noqa: E402
import preprocess  # noqa: E402
from features import window_features  # noqa: E402
from preprocess import CFG, ROOT  # noqa: E402

OUT = ROOT / "results" / "demo"
NAMES = {0: "not fall", 1: "FALL", 2: "lying"}
COLORS = {0: "#c9c8c2", 1: "#eb6834", 2: "#2a78d6"}       # neutral / orange / blue


def available_algos(only=None):
    algos = {}
    for path in sorted((ROOT / "src").glob("run_*.py")):
        algo = path.stem[4:]
        if only and algo not in only:
            continue
        if not list((ROOT / "models").glob(f"{algo}.*")):
            continue
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            algos[algo] = mod.predict_proba
        except Exception as e:                                    # one broken algo must not stop the demo
            print(f"skip {algo}: {e}")
    return algos


def video_windows(poses):
    """Windows of one pose table (all windows, labelled or not)."""
    if "y" not in poses and "label" not in poses:
        poses = poses.assign(y=np.nan)
    prep = preprocess.prepare_sequence(poses)
    meta, X, _ = preprocess.sequence_windows(prep, keep_unlabelled=True)
    meta = pd.DataFrame(meta)
    idx = pd.Index(prep["frames"]).get_indexer(meta["start_frame"])
    feats = window_features(prep, idx)
    feats.insert(0, "window_id", np.arange(len(meta)))
    return meta, feats, X


def run(poses, algos, tau):
    meta, feats, X = video_windows(poses)
    fps = CFG["data"]["fps"]
    table = pd.DataFrame({"end_frame": meta["end_frame"], "time_s": meta["end_frame"] / fps,
                          "y_true": meta["y"]})
    for algo, fn in algos.items():
        p = np.asarray(fn(feats, X), dtype=float)
        table[f"{algo}_p0"], table[f"{algo}_p1"], table[f"{algo}_p2"] = p.T
        table[f"{algo}_class"] = p.argmax(axis=1)
        table[f"{algo}_alert"] = p[:, 1] >= tau
    return table


def summary(table, algos, tau):
    stride = CFG["window"]["stride"] / CFG["data"]["fps"]
    lines = []
    for algo in algos:
        alerts = table.loc[table[f"{algo}_alert"], "time_s"]
        first = f"first alert at {alerts.iloc[0]:.1f} s" if len(alerts) else "no alert"
        lying = (table[f"{algo}_class"] == 2).sum() * stride
        lines.append(f"{algo:>10}: {first}; {table[f'{algo}_alert'].sum()} alerted windows "
                     f"(p1 >= {tau}); ~{lying:.1f} s predicted lying")
    return "\n".join(lines)


def plot(table, algos, tau, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows = (["truth"] if table["y_true"].notna().any() else []) + list(algos)
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 0.9 + 1.1 * len(rows)), sharex=True,
                             squeeze=False)
    half = CFG["window"]["stride"] / CFG["data"]["fps"] / 2
    t = table["time_s"].to_numpy()
    for ax, row in zip(axes[:, 0], rows):
        cls = table["y_true"] if row == "truth" else table[f"{row}_class"]
        for ti, c in zip(t, cls):
            if not np.isnan(c):
                ax.axvspan(ti - half, ti + half, color=COLORS[int(c)], alpha=0.45, lw=0)
        if row != "truth":
            ax.plot(t, table[f"{row}_p1"], color="#0b0b0b", lw=1.5)
            ax.axhline(tau, color="#52514e", lw=0.8, ls="--")
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 1] if row != "truth" else [])
        ax.set_ylabel(row, rotation=0, ha="right", va="center", color="#52514e")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[-1, 0].set_xlabel("time (s), end of each 1 s window")
    handles = [Patch(color=COLORS[k], alpha=0.45, label=NAMES[k]) for k in NAMES]
    handles.append(plt.Line2D([], [], color="#0b0b0b", lw=1.5, label="p1 (fall)"))
    fig.legend(handles=handles, loc="upper right", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("--poses", help="pose parquet (skips YOLO)")
    ap.add_argument("--labels", help="interval labels, e.g. 'Falling[3 to 4]; Lying[4 to 9]'")
    ap.add_argument("--algos", help="comma-separated subset")
    ap.add_argument("--tau", type=float, default=CFG["eval"]["tau"])
    ap.add_argument("--crop", choices=["full", "right"])
    args = ap.parse_args()
    if not args.video and not args.poses:
        ap.error("give a video or --poses")

    if args.poses:
        poses = pd.read_parquet(args.poses)
        name = Path(args.poses).stem
    else:
        import extract_pose
        name = Path(args.video).stem
        crop = args.crop or ("right" if name.endswith("-cam0") else "full")
        poses = extract_pose.extract_video(args.video, seq=name, crop=crop)
    print(f"person detected on {poses['person'].mean():.0%} of frames "
          "(on a clip of someone standing or walking, well below ~90 % means bad framing or light;"
          " a person lying on the floor is often missed)")
    if args.labels:
        poses = poses.drop(columns=["y", "label"], errors="ignore")
        poses["y"], _ = external.frame_labels(external.parse_classes(args.labels), len(poses))

    algos = available_algos(set(args.algos.split(",")) if args.algos else None)
    if not algos:
        sys.exit("no algorithm with both src/run_<algo>.py and models/<algo>.*")
    table = run(poses, algos, args.tau)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / f"{name}.csv", index=False)
    plot(table, algos, args.tau, f"{name}: predicted class and p1 per algorithm", OUT / f"{name}.png")
    view = table[["time_s"] + (["y_true"] if table["y_true"].notna().any() else [])
                 + [f"{a}_class" for a in algos]].copy()
    print(view.to_string(index=False, float_format="{:.2f}".format, max_rows=60))
    print(summary(table, algos, args.tau))
    print(f"-> {OUT / name}.csv, .png")


if __name__ == "__main__":
    main()
