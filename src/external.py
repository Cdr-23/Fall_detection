"""External test sets (GMDCSA-24, the team's own videos): poses + labels + processed windows.

These sets are ONLY for testing: never tune anything on them.

Label rule (interval labels such as "Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]", in seconds):
  - Falling / Fall [a to b]  -> class 1 (fall) for the first second [a, a + 1 s), like the
    30-frame fall phase of URFD, then class 2 (lying) until b. In GMDCSA-24 "Falling" runs to the
    end of the clip (fall + time on the floor), so this split is an approximation; the event
    metrics (fall start s0 = a) do not depend on it.
  - Sleeping / Lying         -> class 2 (lying).
  - any other activity       -> class 0 (not fall) (sitting, walking, standing, reading, ...).
  - frames in no interval    -> NaN (ignored). Fall intervals win over overlapping ones.
GMDCSA-24 label typos: s4-fall-09 'Falling (BW)[2.3 6]' is read as [2.3 to 6];
s4-fall-15 'Falling (FW)' has no interval, so this clip has no fall event (not guessed);
s2-adl-06 'Sitting[9.7 to 3]' is reversed and dropped.
Frame i (1-based) is at time (i - 1) / 30 s, after resampling the poses to 30 fps.

Outputs: poses/<name>/<seq>.parquet (pose columns + y + activity) and processed/<name>/
(windows_meta with fold = -1, features, X_seq, valid, sequences.csv), same format as URFD.

Usage: python src/external.py gmdcsa24     # needs data/raw/gmdcsa24 unzipped (CLAUDE.md, 3)
       python src/external.py le2i         # Le2i FDD Home_01 + Home_02 in data/raw/le2i (Kaggle)
       python src/external.py own          # data/own/*.mp4 + data/own/labels.csv (file, classes)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import extract_pose
import features
import preprocess
from preprocess import CFG, ROOT

RAW = ROOT / "data" / "raw"
PATTERN = re.compile(r"([A-Za-z][A-Za-z ()/_-]*?)\s*\[([^\]]*)\]")
SPAN = re.compile(r"([\d.]+)\s*(?:to)?\s*([\d.]+)")


def parse_classes(text, warn=None):
    """'Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]' -> [('Falling (SW)', 3.4, 6.0), ...].
    Tolerates the typos found in GMDCSA-24: several spans in one bracket ('[0 to 0.5; 4 to 6]'),
    a missing 'to' ('[2.3 6]'). Reversed spans ('[9.7 to 3]') are dropped (reported to `warn`)."""
    out = []
    for name, inside in PATTERN.findall(str(text)):
        for a, b in SPAN.findall(inside):
            a, b = float(a), float(b)
            if a > b:
                if warn:
                    warn(f"dropped reversed span {name.strip()}[{a} to {b}]")
                continue
            out.append((name.strip(), a, b))
    return out


def class_of(name):
    n = name.lower()
    if n.startswith("fall"):
        return 1
    if n.startswith(("sleep", "lying", "lie")):
        return 2
    return 0


def frame_labels(intervals, n_frames, fps=CFG["data"]["fps"]):
    """Per-frame class (float, NaN = unlabelled) and activity name, from interval labels."""
    t = np.arange(n_frames) / fps
    y = np.full(n_frames, np.nan)
    act = np.full(n_frames, "", dtype=object)
    for name, a, b in sorted(intervals, key=lambda x: class_of(x[0]) == 1):   # falls last
        inside = (t >= a - 1e-9) & (t <= b + 1e-9)
        c = class_of(name)
        if c == 1:
            y[inside] = np.where(t[inside] < a + 1.0, 1, 2)
        else:
            y[inside] = c
        act[inside] = name
    return y, act


def gmdcsa_clips(root=RAW / "gmdcsa24"):
    """(seq, video path, classes text) for every clip listed in the Fall.csv / ADL.csv files."""
    clips = []
    for csv in sorted(Path(root).glob("*/Subject */*.csv")):
        subject = csv.parent.name.split()[-1]
        kind = csv.stem.lower()                                   # fall or adl
        for line in csv.read_text(encoding="utf-8-sig").splitlines()[1:]:
            if not line.strip():
                continue
            fname = line.split(",")[0].strip()
            video = csv.parent / csv.stem / fname
            seq = f"gmdcsa-s{subject}-{kind}-{Path(fname).stem}"
            clips.append((seq, video, line))
    return clips


def le2i_clips(root=RAW / "le2i", src_fps=24.0):
    """Le2i FDD (Home_01, Home_02). Annotation file: line 1 = first frame of the fall, line 2 =
    last frame (both 0 = no fall, ADL clip), then one line per frame (frame, posture code, box;
    the posture code is not documented here and is not used). Converted to the interval labels
    used above, with the same rule as GMDCSA-24: not fall before the fall start, then fall for
    1 s, then lying until the end of the clip (approximation; event metrics only use the start).
    Reads the .mp4 made by src/le2i_convert.py (OpenCV crashes on the raw AVIs)."""
    clips = []
    for video in sorted(Path(root).glob("Home_0*/Videos/*.avi")):
        num = int(re.search(r"\((\d+)\)", video.name).group(1))
        ann = video.parent.parent / "Annotation_files" / f"{video.stem}.txt"
        lines = ann.read_text().split()
        start, n = int(lines[0]), len(lines) - 2
        end_t = n / src_fps
        home = video.parent.parent.name[-1]
        if start > 0:
            t0 = (start - 1) / src_fps                        # frames are 1-based
            classes, kind = f"Walking[0 to {t0:.3f}]; Falling[{t0:.3f} to {end_t:.3f}]", "fall"
        else:
            classes, kind = f"Walking[0 to {end_t:.3f}]", "adl"
        clips.append((f"le2i-h{home}-{kind}-{num:02d}", video.with_suffix(".mp4"), classes))
    return clips


def own_clips(root=ROOT / "data" / "own"):
    root = Path(root)
    labels = pd.read_csv(root / "labels.csv", skipinitialspace=True)
    return [(f"own-{Path(f).stem}", root / f, c) for f, c in zip(labels["file"], labels["classes"])]


def extract(clips, poses_dir):
    poses_dir = Path(poses_dir)
    poses_dir.mkdir(parents=True, exist_ok=True)
    model, dev = None, extract_pose.device()
    for seq, video, classes in clips:
        out = poses_dir / f"{seq}.parquet"
        if out.exists():                                          # cached poses: refresh labels only
            df = pd.read_parquet(out)
            y, act = frame_labels(parse_classes(classes), len(df))
            if not (np.array_equal(df["y"].to_numpy(), y, equal_nan=True)
                    and (df["activity"].to_numpy() == act).all()):
                df["y"], df["activity"] = y, act
                df.to_parquet(out)
                print(f"{seq}: labels updated")
            continue
        if not Path(video).exists():
            print(f"{seq}: missing video {video}")
            continue
        model = model or extract_pose.load_model()
        df = extract_pose.extract_video(video, model, dev, seq=seq)
        df["y"], df["activity"] = frame_labels(parse_classes(classes, lambda m: print(seq, m)),
                                               len(df))
        df.to_parquet(out)
        print(f"{seq}: {len(df)} frames @30fps (src {df.src_fps.iloc[0]:.0f} fps), "
              f"person {df.person.mean():.0%}, classes {np.unique(df.y[~np.isnan(df.y)])}",
              flush=True)


def build(name):
    clips = {"gmdcsa24": gmdcsa_clips, "le2i": le2i_clips}.get(name, own_clips)()
    poses_dir, out = ROOT / "poses" / name, ROOT / "processed" / name
    extract(clips, poses_dir)
    meta = preprocess.build(poses_dir=poses_dir, out=out, cv=False)
    (out / "folds.csv").unlink(missing_ok=True)                 # no split on a test set
    features.build(processed=out, poses_dir=poses_dir)
    print(f"{len(meta)} windows from {meta.seq.nunique()} sequences")
    print(pd.crosstab(meta["kind"], meta["y"].map(preprocess.CLASSES), margins=True))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "gmdcsa24")
