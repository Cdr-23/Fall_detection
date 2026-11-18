"""Longer confirmation: alert only if the person then stays on the floor for D seconds.

Extends src/confirm.py (P3): a fall candidate (p1 >= 0.5) is confirmed when, within 3 s after it,
a run of contiguous windows classified lying (p2 >= 0.5) lasts D seconds. Two variants:
  strict   : every window of the run must be classified lying;
  lenient  : a window with no person detected (valid_frac < 0.5) also counts as "on the floor"
             (YOLO misses ~35 % of lying frames), at the risk of confirming someone who left.

Limit of the data: URFD stops filming 1.1-3.1 s after the fall starts and GMDCSA-24 1.4-9.1 s,
so D cannot be tested beyond ~3 s on falls, and 10-30 s not at all. Falls are therefore scored
only on the "testable" falls (filmed >= 1 s + D after the start); false alerts on all ADL clips
(URFD ~8 s each, GMDCSA ~9 s each).

Longer delays (10, 20, 30 s: the product setting we recommend) are only scored on the team videos
(data/own, processed/own), which must keep the person on the floor >= 30 s after each fall and
last several minutes for the ADL; on the public sets the clips end before.

URFD: out-of-fold predictions. GMDCSA-24, Le2i, own: final models (models/*), never tuned on them.
Output: results/confirm_long.csv. Usage: python src/confirm_long.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from confirm import confirm_alerts  # noqa: E402
from evaluate import load_predict_proba  # noqa: E402
from metrics import CFG, ROOT, event_metrics  # noqa: E402

PROCESSED, RESULTS = ROOT / "processed", ROOT / "results"
ALGOS = ["logreg", "cart", "boosting", "nn"]
DURATIONS = [0, 0.5, 1, 2, 3, 5]          # seconds on the floor; 0 = raw alert, no confirmation
LONG = [10, 20, 30]                       # product setting, only on the team videos (data/own)
FPS, STRIDE, TOL = CFG["data"]["fps"], CFG["window"]["stride"], CFG["eval"]["event_tolerance"]


def predictions(dataset):
    p = PROCESSED if dataset == "urfd" else PROCESSED / dataset
    meta = pd.read_parquet(p / "windows_meta.parquet")
    seqs = pd.read_csv(p / "sequences.csv")
    out = {}
    for a in ALGOS:
        if dataset == "urfd":
            oof = pd.read_parquet(RESULTS / f"{a}_oof.parquet")
        else:
            proba = load_predict_proba(a)(pd.read_parquet(p / "features.parquet"), np.load(p / "X_seq.npy"))
            oof = meta[["window_id", "seq", "end_frame"]].assign(
                fold=0, y_true=meta["y"], p0=proba[:, 0], p1=proba[:, 1], p2=proba[:, 2],
                y_pred=proba.argmax(axis=1))
        vf = oof["window_id"].map(meta.set_index("window_id")["valid_frac"])
        out[a] = (oof.reset_index(drop=True), vf.to_numpy())
    return out, seqs


def score(oof, seqs, d, testable):
    if d == 0:
        alert, tol = oof["p1"].to_numpy() >= 0.5, TOL
    else:
        n = max(1, round(d * FPS / STRIDE))
        horizon = 3 * FPS + n * STRIDE
        alert, tol = confirm_alerts(oof, n_lying=n, horizon=horizon), TOL + horizon
    m = event_metrics(oof.assign(p1=alert.astype(float)), seqs, tau=0.5, tol=tol)
    ev = event_metrics(oof[oof["seq"].isin(testable) | ~oof["seq"].str.contains("fall")]
                       .assign(p1=alert[oof["seq"].isin(testable) | ~oof["seq"].str.contains("fall")]
                               .astype(float)), seqs, tau=0.5, tol=tol)
    return {"falls_detected": ev["n_detected"], "falls_testable": ev["n_falls"],
            "delay_mean_s": ev["delay_mean_s"], "false_alerts_adl": m["false_alerts_adl"],
            "adl_clips_alerted": m["adl_seqs_with_false_alert"]}


def main():
    rows = []
    datasets = ["urfd", "gmdcsa24", "le2i"]
    if (PROCESSED / "own" / "windows_meta.parquet").exists():
        datasets.append("own")                # built by: python src/external.py own
    for dataset in datasets:
        preds, seqs = predictions(dataset)
        falls = seqs.dropna(subset=["fall_start"])
        after = (falls["n_frames"] - falls["fall_start"]) / FPS
        n_adl = int((seqs["kind"] == "adl").sum())
        for d in DURATIONS + (LONG if dataset == "own" else []):
            testable = set(falls.loc[after >= 1 + d, "seq"])
            for a, (oof, vf) in preds.items():
                for variant in (["strict", "lenient"] if d else ["raw"]):
                    o = oof if variant != "lenient" else oof.assign(p2=np.where(vf < 0.5, 1.0, oof["p2"]))
                    rows.append({"dataset": dataset, "algo": a, "seconds_on_floor": d,
                                 "variant": variant, "adl_clips": n_adl, **score(o, seqs, d, testable)})
        print(dataset, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "confirm_long.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 200, "display.float_format", "{:.2f}".format):
        print(res.to_string(index=False))
    print("->", RESULTS / "confirm_long.csv")


if __name__ == "__main__":
    main()
