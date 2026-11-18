"""Convert the Le2i FDD videos to MP4 so that the shared pipeline can read them.

The Le2i AVIs store raw, uncompressed 24-bit bitmaps (codec 'DIB '), which make OpenCV crash on
this machine (bus error while decoding). This script reads the frames directly from the AVI
chunks ('00db' / '00dc' in the 'movi' list), without any decoder, and writes them to MP4 with
OpenCV at the source frame rate (24 fps). Audio is dropped.

Input: data/raw/le2i/Home_0*/Videos/*.avi. Output: same folder, .mp4 next to each .avi.
Usage: python src/le2i_convert.py
"""
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "le2i"


def avi_frames(path):
    """Yield BGR frames (H, W, 3) of an uncompressed 24-bit AVI, plus the fps (first item)."""
    data = Path(path).read_bytes()
    j = data.find(b"strf")
    w, h, _, bits = struct.unpack("<iiHH", data[j + 12:j + 24])
    assert bits == 24, f"{path}: {bits}-bit frames not supported"
    i = data.find(b"strh")
    scale, rate = struct.unpack("<II", data[i + 28:i + 36])
    yield rate / scale
    row = (w * 3 + 3) // 4 * 4                        # bitmap rows are padded to 4 bytes
    pos = data.find(b"movi") + 4
    end = len(data)
    while pos + 8 <= end:
        tag, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if tag == b"LIST":
            pos += 12
            continue
        if tag[2:] in (b"db", b"dc") and size == row * abs(h):
            img = np.frombuffer(data, np.uint8, row * abs(h), pos + 8).reshape(abs(h), row)
            img = img[:, :w * 3].reshape(abs(h), w, 3)
            yield img[::-1] if h > 0 else img             # positive height = bottom-up bitmap
        elif tag == b"idx1":
            break
        pos += 8 + size + (size & 1)


def convert(avi):
    out = avi.with_suffix(".mp4")
    if out.exists():
        return out, None
    frames = avi_frames(avi)
    fps = next(frames)
    writer, n = None, 0
    for img in frames:
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (img.shape[1], img.shape[0]))
        writer.write(np.ascontiguousarray(img))
        n += 1
    writer.release()
    return out, n


def main():
    videos = sorted(RAW.glob("Home_0*/Videos/*.avi"))
    if not videos:
        sys.exit(f"no AVI in {RAW}")
    for avi in videos:
        out, n = convert(avi)
        print(f"{avi.parent.parent.name}/{avi.name} -> {out.name}" + (f" ({n} frames)" if n else " (exists)"),
              flush=True)


if __name__ == "__main__":
    main()
