"""Alert confirmation by lying: alert only on a fall FOLLOWED by the person lying (P3 proposal).

Why: at tau = 0.5 most false alerts are confident (p1 ~ 0.85) and come from fast sitting or
bending, which look like a fall within one second of skeleton. What separates them from a real
fall is what happens next: after a fall the person stays on the floor. For the BSPP, waiting
2-3 s more is negligible next to a time on the floor counted in hours.

Rule (applied per sequence, on the windows in time order; works for any algorithm's OOF):
  - candidate  = a window with p1 >= tau;
  - confirmed  = the first window w such that the last N_LYING consecutive windows up to w
                 (no gap) all have p2 >= LYING_TAU, and a candidate ended at most HORIZON
                 frames before w (and not after it).
  The alert is raised at the confirmed window only. Parameters are fixed a priori (below),
  not tuned on the folds; the sensitivity table varies them for information only.

Scoring: the shared event metrics of metrics.py, with the detection tolerance extended by
HORIZON (a confirmed alert comes later by construction): a fall is detected if a confirmed
alert ends in [s0, s0 + 44 + HORIZON]. Window-level metrics are unchanged (same p0/p1/p2).
Limits: a voluntary lying down (bed, sofa) preceded by a fall-like motion is still alerted;
YOLO misses ~35 % of lying frames, which can delay or prevent the confirmation.

Outputs: results/confirm_summary.csv (raw vs confirmed alerts, every results/*_oof.parquet),
results/confirm_sensitivity.csv. Usage: python src/confirm.py
"""
import numpy as np
import pandas as pd

from common import RESULTS
from metrics import CFG, event_metrics
from preprocess import PROCESSED

TAU = CFG["eval"]["tau"]          # candidate threshold on p1 (same default as everyone)
LYING_TAU = 0.5                   # threshold on p2 for a window to count as lying
N_LYING = 3                       # consecutive lying windows needed (~0.5 s at stride 5)
HORIZON = 90                      # frames (3 s) allowed between the fall and the confirmation
STRIDE = CFG["window"]["stride"]
TOL = CFG["eval"]["event_tolerance"]


def confirm_alerts(oof, tau=TAU, lying_tau=LYING_TAU, n_lying=N_LYING, horizon=HORIZON):
    """Boolean array aligned with `oof` rows: True on windows raising a confirmed alert."""
    out = np.zeros(len(oof), bool)
    for _, g in oof.groupby("seq", sort=False):
        g = g.sort_values("end_frame")
        end = g["end_frame"].to_numpy()
        cand = g["p1"].to_numpy() >= tau
        lying = g["p2"].to_numpy() >= lying_tau
        run = np.zeros(len(g), int)  # length of the current run of contiguous lying windows
        for i in range(len(g)):
            contiguous = i > 0 and end[i] - end[i - 1] == STRIDE
            run[i] = (run[i - 1] + 1 if contiguous else 1) if lying[i] else 0
        cand_end = end[cand]
        for i in np.flatnonzero(run >= n_lying):
            if ((cand_end <= end[i]) & (cand_end >= end[i] - horizon)).any():
                out[oof.index.get_indexer([g.index[i]])[0]] = True
    return out


def confirmed_events(oof, sequences, **kw):
    """Shared event metrics on confirmed alerts (tolerance extended by the horizon)."""
    horizon = kw.get("horizon", HORIZON)
    oof = oof.reset_index(drop=True)
    alerted = oof.assign(p1=confirm_alerts(oof, **kw).astype(float))
    return event_metrics(alerted, sequences, tau=0.5, tol=TOL + horizon)


KEYS = ["n_detected", "n_falls", "event_recall", "delay_mean_s", "false_alerts_adl",
        "adl_seqs_with_false_alert", "false_alerts_fall_seqs"]


def main():
    sequences = pd.read_csv(PROCESSED / "sequences.csv")
    rows, sens = [], []
    for path in sorted(RESULTS.glob("*_oof.parquet")):
        algo = path.stem.removesuffix("_oof")
        oof = pd.read_parquet(path)
        raw = event_metrics(oof, sequences, tau=TAU)
        conf = confirmed_events(oof, sequences)
        rows += [{"algo": algo, "alert": "raw (p1 >= tau)", **{k: raw[k] for k in KEYS}},
                 {"algo": algo, "alert": "confirmed by lying", **{k: conf[k] for k in KEYS}}]
        for h in (60, 90, 150):
            for n in (1, 3, 6):
                m = confirmed_events(oof, sequences, horizon=h, n_lying=n)
                sens.append({"algo": algo, "horizon_frames": h, "n_lying": n,
                             **{k: m[k] for k in KEYS}})
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "confirm_summary.csv", index=False)
    pd.DataFrame(sens).to_csv(RESULTS / "confirm_sensitivity.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(summary.to_string(index=False))
    print("->", RESULTS / "confirm_summary.csv", RESULTS / "confirm_sensitivity.csv")


if __name__ == "__main__":
    main()
