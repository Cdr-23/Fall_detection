"""Checks of the context features (P7, src/features_ctx.py). Descriptive only: nothing is
changed according to what they show.

1. Hand recomputation: for 3 URFD sequences drawn at random (seed 42), one window each, one
   feature per group recomputed with plain loops (independent of features_ctx.py):
   hip_above_floor_mean (FLOOR), descent_time_3s (3 s look-back), torso_angle_mean_lag1s
   (previous windows; also compared with the base feature of the window ending 30 frames earlier).
2. Causality of the feature code: every prepared per-frame array (kp, bbox, detected) is
   overwritten with noise after the window's last frame; the features of every window ending
   at or before it must not change (scale kept, as it is a per-sequence constant).
3. Causality end to end: the raw poses after the last frame are replaced by noise, then the
   shared preprocess (interpolation of gaps <= 15 frames) is rerun with the same scale. Any
   change here comes from the shared pipeline (it also affects the 26 base features).
4. Orders of magnitude: hip_above_floor_mean / head_above_floor_mean of upright windows (class
   not fall, torso_angle_mean < 30), lying windows of fall-* and of adl-* sequences
   -> results/p7_floor_check.png.
5. Share of NaN per feature and class -> results/p7_nan_share.csv.

Text summary -> results/p7_feature_checks.md. Usage: python src/check_features_ctx.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import features  # noqa: E402
import features_ctx as fc  # noqa: E402
from preprocess import CFG, CLASSES, PROCESSED, ROOT, load_poses, prepare_sequence  # noqa: E402

RESULTS = ROOT / "results"
SEED = CFG["seed"]


def md(df):
    """Minimal Markdown table (avoids the optional `tabulate` dependency)."""
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def hand_features(prep, s, length=30):
    """Plain-loop recomputation of one feature per group for the window starting at s."""
    kp, scale, det = prep["kp"], prep["scale"], prep["detected"]
    e = s + length - 1
    T = len(kp)

    def hip_y(t):
        v = [kp[t, j, 1] for j in (11, 12) if not np.isnan(kp[t, j, 1])]
        return np.mean(v) if v else np.nan

    def torso_angle(t):
        hx = [kp[t, j, 0] for j in (11, 12) if not np.isnan(kp[t, j, 0])]
        hy = [kp[t, j, 1] for j in (11, 12) if not np.isnan(kp[t, j, 1])]
        sx = [kp[t, j, 0] for j in (5, 6) if not np.isnan(kp[t, j, 0])]
        sy = [kp[t, j, 1] for j in (5, 6) if not np.isnan(kp[t, j, 1])]
        if not (hx and sx):
            return np.nan
        return np.degrees(np.arctan2(abs(np.mean(sx) - np.mean(hx)), -(np.mean(sy) - np.mean(hy))))

    # FLOOR: hip_above_floor_mean
    vals = []
    for t in range(s, e + 1):
        standing = [u for u in range(t + 1) if det[u] and torso_angle(u) < 30
                    and not np.isnan(kp[u, 15, 1]) and not np.isnan(kp[u, 16, 1])]
        if len(standing) < 10 or np.isnan(hip_y(t)):
            continue
        ref = np.median([(kp[u, 15, 1] + kp[u, 16, 1]) / 2 for u in standing[-30:]])
        vals.append((ref - hip_y(t)) / scale)
    hip_floor = np.mean(vals) if vals else np.nan
    # 3 s: descent_time_3s
    b = max(0, e - 89)
    hy = [(t, hip_y(t)) for t in range(b, e + 1) if not np.isnan(hip_y(t))]
    descent = (prep["frames"][e] - prep["frames"][min(hy, key=lambda x: x[1])[0]]) / 30 if hy else np.nan
    # previous window: torso_angle_mean_lag1s
    ls, le = e - 59, e - 30
    lag = np.nanmean([torso_angle(t) for t in range(ls, le + 1)]) if ls >= 0 else np.nan
    assert T > e
    return {"hip_above_floor_mean": hip_floor, "descent_time_3s": descent,
            "torso_angle_mean_lag1s": lag}


def perturb_prep(prep, e, rng):
    p = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in prep.items()}
    n = len(p["kp"]) - e - 1
    p["kp"][e + 1:] = rng.uniform(0, 320, (n, 17, 2))
    p["bbox"][e + 1:] = rng.uniform(0, 320, (n, 4))
    p["detected"][e + 1:] = rng.random(n) < 0.5
    return p


def perturb_raw(df, end_frame, rng):
    df = df.sort_values("frame").reset_index(drop=True).copy()
    after = df["frame"] > end_frame
    n = int(after.sum())
    for j in range(17):
        df.loc[after, f"kp{j}_x"] = rng.uniform(0, 320, n).astype(df[f"kp{j}_x"].dtype)
        df.loc[after, f"kp{j}_y"] = rng.uniform(0, 240, n).astype(df[f"kp{j}_y"].dtype)
        df.loc[after, f"kp{j}_c"] = 1.0
    df.loc[after, "person"] = 1
    return df


def main():
    rng = np.random.default_rng(SEED)
    meta = pd.read_parquet(PROCESSED / "windows_meta.parquet")
    fx = pd.read_parquet(PROCESSED / "features_ctx.parquet")
    base = pd.read_parquet(PROCESSED / "features.parquet")
    lines = ["# P7 feature checks (generated by src/check_features_ctx.py)", ""]

    # 1-3 on 3 random sequences (windows far enough from the start to have a full history)
    seqs = rng.choice(sorted(meta["seq"].unique()), 3, replace=False)
    lines += ["## 1. Hand recomputation (stored value / loop value)", "",
              "| Sequence | Window end | hip_above_floor_mean | descent_time_3s | "
              "torso_angle_mean_lag1s | base torso_angle_mean of the window 30 frames earlier |",
              "|---|---|---|---|---|---|"]
    caus = []
    for seq in seqs:
        m = meta[meta["seq"] == seq]
        prep = prepare_sequence(load_poses(seq))
        cand = m[m["end_frame"] >= 90]
        row = cand.iloc[rng.integers(len(cand))] if len(cand) else m.iloc[-1]
        s = int(pd.Index(prep["frames"]).get_loc(row["start_frame"]))
        e = s + 29
        hand = hand_features(prep, s)
        stored = fx.set_index("window_id").loc[row["window_id"]]
        prev = m[m["end_frame"] == row["end_frame"] - 30]
        prev_txt = (f"{base.set_index('window_id').loc[prev['window_id'].iloc[0], 'torso_angle_mean']:.4f}"
                    if len(prev) else "n/a")
        lines.append(f"| {seq} | {int(row['end_frame'])} | "
                     + " | ".join(f"{stored[k]:.4f} / {hand[k]:.4f}" for k in hand)
                     + f" | {prev_txt} |")
        for k in hand:
            assert np.isclose(stored[k], hand[k], equal_nan=True), (seq, k, stored[k], hand[k])

        # 2. causality of the feature code: every window ending <= e
        starts = np.arange(0, s + 1, 5)
        ref = fc.window_features_ctx(prep, starts)
        pert = fc.window_features_ctx(perturb_prep(prep, e, rng), starts)
        code_ok = np.array_equal(ref.to_numpy(), pert.to_numpy(), equal_nan=True)
        assert code_ok, seq
        # 3. end to end, through the shared preprocess (same scale)
        prep2 = prepare_sequence(perturb_raw(load_poses(seq), int(prep["frames"][e]), rng))
        prep2["scale"] = prep["scale"]
        pert2 = fc.window_features_ctx(prep2, starts)
        diff = ~np.isclose(ref.to_numpy(), pert2.to_numpy(), equal_nan=True)
        changed = pd.Series(diff.any(axis=0), index=ref.columns)
        caus.append({"seq": seq, "end_frame": int(prep["frames"][e]), "windows": len(starts),
                     "code_level_identical": code_ok,
                     "e2e_windows_changed": int(diff.any(axis=1).sum()),
                     "e2e_base_features_changed": int(changed[features.FEATURES].sum()),
                     "e2e_ctx_features_changed": int(changed[fc.CTX].sum())})
    lines += ["", "All three values of each group match (np.isclose).", "",
              "## 2-3. Causality (windows ending at or before the window above)", "",
              md(pd.DataFrame(caus)), "",
              "code_level_identical: the feature code reads no frame after the window end. "
              "e2e_*: changes caused by the shared preprocess, which interpolates gaps of at most "
              "15 frames (0.5 s) using the next valid frame; it affects base and context "
              "features alike.", ""]

    # 4. orders of magnitude
    d = meta.merge(fx, on="window_id")
    groups = {"upright (not fall,\ntorso angle < 30)": (d["y"] == 0) & (d["torso_angle_mean"] < 30),
              "lying, fall-*": (d["y"] == 2) & (d["kind"] == "fall"),
              "lying, adl-*": (d["y"] == 2) & (d["kind"] == "adl")}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    stats = []
    for ax, f in zip(axes, ["hip_above_floor_mean", "head_above_floor_mean"]):
        data = [d.loc[g, f].dropna().to_numpy() for g in groups.values()]
        ax.boxplot(data, tick_labels=list(groups), showfliers=True)
        ax.axhline(0, color="grey", lw=0.8, ls=":")
        ax.set(title=f, ylabel="torso lengths above the floor reference")
        for name, g in groups.items():
            v = d.loc[g, f]
            stats.append({"feature": f, "group": name.replace("\n", " "), "windows": int(g.sum()),
                          "non_nan": int(v.notna().sum()), "median": v.median(),
                          "q25": v.quantile(0.25), "q75": v.quantile(0.75)})
    fig.suptitle("URFD: height above the floor (floor = median mid-ankle y of earlier standing frames)")
    fig.tight_layout()
    fig.savefig(RESULTS / "p7_floor_check.png", dpi=130)
    plt.close(fig)
    lines += ["## 4. Orders of magnitude (URFD) -> results/p7_floor_check.png", "",
              md(pd.DataFrame(stats).round(2)), ""]

    # 5. NaN share per feature and class, on the three sets
    nan_rows = []
    for name, (proc, _) in fc.SETS.items():
        m = pd.read_parquet(proc / "windows_meta.parquet")
        f = pd.read_parquet(proc / "features_ctx.parquet")
        for c, cname in CLASSES.items():
            sel = (m["y"] == c).to_numpy()
            for col in fc.CTX:
                nan_rows.append({"set": name, "class": cname, "feature": col,
                                 "windows": int(sel.sum()),
                                 "nan_share": float(f.loc[sel, col].isna().mean())})
    nan = pd.DataFrame(nan_rows)
    nan.to_csv(RESULTS / "p7_nan_share.csv", index=False)
    piv = nan[nan["set"] == "urfd"].pivot(index="feature", columns="class", values="nan_share")
    piv = piv.loc[fc.CTX, ["not fall", "fall", "lying"]]
    lines += ["## 5. Share of NaN per feature and class (URFD; all sets in results/p7_nan_share.csv)",
              "", md((100 * piv).round(0).astype(int).astype(str).add(" %").reset_index()), ""]
    for name in ("gmdcsa24", "le2i"):
        sub = nan[(nan["set"] == name)].pivot(index="feature", columns="class", values="nan_share")
        lines.append(f"{name}, fall class: FLOOR NaN {100 * sub.loc[fc.FLOOR, 'fall'].mean():.0f} %, "
                     f"lag1s NaN {100 * sub.loc[[c for c in fc.CTX if c.endswith('lag1s')], 'fall'].mean():.0f} %, "
                     f"lag2s NaN {100 * sub.loc[[c for c in fc.CTX if c.endswith('lag2s')], 'fall'].mean():.0f} %.")
    text = "\n".join(lines) + "\n"
    (RESULTS / "p7_feature_checks.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
