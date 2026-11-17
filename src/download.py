"""Download the UR Fall Detection dataset (camera 0) as MP4 + per-frame label CSVs, and check it.

Each MP4 is 640x240: depth on the left half, RGB on the right half (320x240).
Source: Kwolek & Kepski (2014), https://fenix.ur.edu.pl/mkepski/ds/uf.html, CC BY-NC-SA 4.0.

Usage: python src/download.py            # URFD (115 MB)
       python src/download.py --gmdcsa   # also GMDCSA-24 (1.1 GB, external test set)
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "params.yaml").read_text())
RAW = ROOT / "data" / "raw"
BASE = CFG["data"]["base_url"]
GMDCSA_URL = ("https://zenodo.org/api/records/13354453/files/ekramalam/"
              "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.1.zip/content")


def urfd_files():
    names = [f"fall-{i:02d}-cam0.mp4" for i in range(1, CFG["data"]["n_falls"] + 1)]
    names += [f"adl-{i:02d}-cam0.mp4" for i in range(1, CFG["data"]["n_adls"] + 1)]
    return names + ["urfall-cam0-falls.csv", "urfall-cam0-adls.csv"]


def fetch(url, dest):
    dest = Path(dest)
    size = int(requests.head(url, allow_redirects=True, timeout=30).headers.get("Content-Length", -1))
    if dest.exists() and (size < 0 or dest.stat().st_size == size):
        return dest.name, "cached"
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if size >= 0 and tmp.stat().st_size != size:
        tmp.unlink()
        return dest.name, "size mismatch"
    tmp.rename(dest)
    return dest.name, "ok"


def check_urfd():
    """Every video must open and contain at least as many frames as the last labelled frame."""
    cols = ["seq", "frame", "label"]
    labels = pd.concat([pd.read_csv(RAW / f, header=None, usecols=[0, 1, 2], names=cols)
                        for f in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv")])
    last = labels.groupby("seq").frame.max()
    bad, total = [], 0
    for seq, max_frame in last.items():
        cap = cv2.VideoCapture(str(RAW / f"{seq}-cam0.mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total += n
        if n < max_frame:
            bad.append(seq)
    print(f"{len(last)} sequences, {total} frames, {len(bad)} with missing frames")
    return not bad and len(last) == CFG["data"]["n_falls"] + CFG["data"]["n_adls"]


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    jobs = [(f"{BASE}/{n}", RAW / n) for n in urfd_files()]
    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(lambda j: fetch(*j), jobs))
    failed = [n for n, s in results if s not in ("ok", "cached")]
    if failed:
        sys.exit(f"download failed: {failed}")
    if "--gmdcsa" in sys.argv:
        (RAW / "gmdcsa24").mkdir(exist_ok=True)
        print(*fetch(GMDCSA_URL, RAW / "gmdcsa24" / "GMDCSA24-v2.1.zip"))
    if not check_urfd():
        sys.exit("URFD check failed")
    print("URFD OK")


if __name__ == "__main__":
    main()
