"""P5 - Final comparison of every algorithm (CLAUDE.md sections 7.5 and 9).

All metrics are recomputed here with src/metrics.py from results/<algo>_oof.parquet (not the
authors' numbers), on the same windows, labels and 5 outer folds.

1. Table at tau = 0.5: results/comparison.csv and results/comparison.md (fold mean +/- std).
2. Figures: results/pr_curves.png, results/confusion_matrices.png, results/tau_tradeoff.png.
3. Alert threshold tau chosen WITHOUT the test fold: for each outer fold k, tau minimises
   R x missed falls + false alerts on the out-of-fold predictions of the 4 other folds, then
   fold k is scored with it. R = cost of a missed fall / cost of a false alert comes from the
   impact model (src/impact.py, ~70); R in {1, 3, 10, 30} for sensitivity
   -> results/tau_choice.csv, results/tau_sensitivity.csv.
4. Errors at tau = 0.5: results/errors.csv (false alerts and missed falls per algo and sequence).
5. External tests: final models (models/*) on GMDCSA-24 and Le2i (processed/<name>), never tuned
   on them, at tau = 0.5 and at the tau chosen on all URFD out-of-fold predictions (R from the impact model):
   results/external_gmdcsa24.csv, results/external_le2i.csv.

Usage: python src/compare.py      (~2 min)
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import cost_rows, load_predict_proba  # noqa: E402
from metrics import ROOT, evaluate, evaluate_by_fold, event_metrics, wilson  # noqa: E402
import impact  # noqa: E402

RESULTS, PROCESSED = ROOT / "results", ROOT / "processed"
ALGOS = ["naive", "rules", "logreg", "cart", "boosting", "nn"]
LABELS = {"naive": "Naive", "rules": "Threshold rules", "logreg": "Logistic regression",
          "cart": "Decision tree", "boosting": "Gradient boosting", "nn": "Neural network"}
TAUS = np.round(np.arange(0.05, 1.0, 0.05), 2)
# R = cost of a missed fall / cost of a false alert. R_IMPACT comes from the impact model
# (config/impact.yaml, ~70); 1-30 are kept for the sensitivity analysis.
R_IMPACT = round(impact.compute({k: impact.CFG[k]["value"] for k in impact.PARAMS})["R_missed_fall_vs_false_alert"])
RATIOS, R_DEFAULT = [1, 3, 10, 30, R_IMPACT], R_IMPACT
EXTERNAL = {"gmdcsa24": "GMDCSA-24", "le2i": "Le2i FDD (Home_01 + Home_02)"}


def load_oof(algo):
    return pd.read_parquet(RESULTS / f"{algo}_oof.parquet")


def ci(k, n):
    lo, hi = wilson(int(k), int(n))
    return f"{max(lo, 0):.0%}-{hi:.0%}"


def pm(table, key, digits=2):
    m, s = table.loc[key, "fold_mean"], table.loc[key, "fold_std"]
    return f"{m:.{digits}f} ± {s:.{digits}f}" if pd.notna(m) else "—"


# ------------------------------------------------------------------ 1. table at tau = 0.5 ----
def comparison(seqs):
    rows, md = [], []
    for a in ALGOS:
        t = evaluate_by_fold(load_oof(a), seqs)
        cost = cost_rows(a)
        v = t["value"]
        rows.append({"algo": a, "pr_auc_fall": t.loc["pr_auc_fall", "fold_mean"],
                     "pr_auc_fall_std": t.loc["pr_auc_fall", "fold_std"],
                     "roc_auc_fall": t.loc["roc_auc_fall", "fold_mean"],
                     "f1_notfall": t.loc["f1_notfall", "fold_mean"],
                     "f1_fall": t.loc["f1_fall", "fold_mean"],
                     "f1_lying": t.loc["f1_lying", "fold_mean"],
                     "falls_detected": int(v["n_detected"]), "n_falls": int(v["n_falls"]),
                     "delay_mean_s": v["delay_mean_s"],
                     "false_alerts_adl": int(v["false_alerts_adl"]),
                     "false_alerts_per_hour": v["false_alerts_per_hour"],
                     "false_alerts_fall_seqs": int(v["false_alerts_fall_seqs"]),
                     "latency_ms": cost.get("latency_ms_per_window"),
                     "size_kb": cost.get("model_size_kb")})
        d = v["delay_mean_s"]
        md.append(f"| {LABELS[a]} | {pm(t, 'pr_auc_fall')} | {pm(t, 'roc_auc_fall')} | "
                  f"{pm(t, 'f1_fall')} | {pm(t, 'f1_lying')} | {int(v['n_detected'])} / "
                  f"{int(v['n_falls'])} ({ci(v['n_detected'], v['n_falls'])}) | {'—' if pd.isna(d) else f'{d:.2f}'} | "
                  f"{int(v['false_alerts_adl'])} ({v['false_alerts_per_hour']:.0f} / h) | "
                  f"{rows[-1]['latency_ms']:.2f} | {rows[-1]['size_kb']:.0f} |")
        print(f"{a}: PR-AUC {pm(t, 'pr_auc_fall')}, {int(v['n_detected'])}/30 falls, "
              f"{int(v['false_alerts_adl'])} false alerts", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "comparison.csv", index=False)
    head = ("| Model | PR-AUC fall | ROC-AUC fall | F1 fall | F1 lying | Falls detected (95 % interval) | "
            "Delay (s) | False alerts ADL (raw, rate) | Latency (ms) | Size (KB) |\n"
            "|---|---|---|---|---|---|---|---|---|---|")
    return head + "\n" + "\n".join(md)


# --------------------------------------------------------------------------- 2. figures ----
def figures(seqs):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for a in ALGOS:
        oof = load_oof(a)
        y, p1 = (oof["y_true"] == 1).to_numpy(), oof["p1"].to_numpy()
        if a == "naive":
            ax.axhline(y.mean(), color="grey", ls=":", label=f"{LABELS[a]} (base rate {y.mean():.2f})")
        elif a == "rules":
            pred = p1 >= 0.5
            ax.plot((pred & y).sum() / y.sum(), (pred & y).sum() / pred.sum(), "ks",
                    label=LABELS[a] + " (0/1 output)")
        else:
            p, r, _ = precision_recall_curve(y, p1)
            ap = evaluate(oof, seqs)["pr_auc_fall"]
            ax.plot(r, p, label=f"{LABELS[a]} (pooled PR-AUC {ap:.2f})")
    ax.set(xlabel="Recall (fall windows)", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02),
           title="Precision / recall of class fall, URFD out-of-fold")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS / "pr_curves.png", dpi=150)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    names = ["not fall", "fall", "lying"]
    for ax, a in zip(axes.flat, ALGOS):
        m = evaluate(load_oof(a), seqs)
        cm = np.array([[m[f"cm_true{i}_pred{j}"] for j in range(3)] for i in range(3)])
        frac = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n{frac[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if frac[i, j] > 0.6 else "black")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=names, yticklabels=names,
               title=LABELS[a], xlabel="predicted (argmax)", ylabel="true")
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrices.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for a in ALGOS[2:]:
        oof = load_oof(a)
        pts = [event_metrics(oof, seqs, t) for t in TAUS]
        fa = [p["false_alerts_adl"] for p in pts]
        miss = [p["n_falls"] - p["n_detected"] for p in pts]
        ax.plot(fa, miss, "o-", ms=3, label=LABELS[a])
        i = list(TAUS).index(0.5)
        ax.plot(fa[i], miss[i], "k*", ms=9)
    r = event_metrics(load_oof("rules"), seqs)
    ax.plot(r["false_alerts_adl"], r["n_falls"] - r["n_detected"], "ks", label="Threshold rules")
    ax.set(xlabel="False alerts in the 40 ADL sequences (raw count, ~4 min of monitoring)",
           ylabel="Missed falls (out of 30)",
           title="Alert threshold trade-off (tau from 0.05 to 0.95; * = tau 0.5)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "tau_tradeoff.png", dpi=150)


# ------------------------------------------------------------- 3. tau chosen per fold ----
def best_tau(oof, seqs, ratio):
    costs = []
    for t in TAUS:
        e = event_metrics(oof, seqs, t)
        missed = e["n_falls"] - e["n_detected"]
        costs.append(ratio * missed + e["false_alerts_adl"] + e["false_alerts_fall_seqs"])
    return float(TAUS[int(np.argmin(costs))])   # ties -> the lowest tau (fewer missed falls)


def tau_choice(seqs):
    rows, chosen = [], []
    for a in ALGOS[1:]:
        oof = load_oof(a)
        for ratio in RATIOS:
            alert = oof.copy()
            taus = {}
            for k in sorted(oof["fold"].unique()):
                taus[k] = best_tau(oof[oof["fold"] != k], seqs, ratio)
                te = alert["fold"] == k
                alert.loc[te, "p1"] = (oof.loc[te, "p1"] >= taus[k]).astype(float)
            e = event_metrics(alert, seqs, 0.5)
            rows.append({"algo": a, "R": ratio, "tau_per_fold": list(taus.values()),
                         "falls_detected": e["n_detected"], "delay_mean_s": e["delay_mean_s"],
                         "false_alerts_adl": e["false_alerts_adl"],
                         "false_alerts_fall_seqs": e["false_alerts_fall_seqs"]})
        chosen.append({"algo": a, "tau_final": best_tau(oof, seqs, R_DEFAULT)})
    sens = pd.DataFrame(rows)
    sens.to_csv(RESULTS / "tau_sensitivity.csv", index=False)
    chosen = pd.DataFrame(chosen)
    chosen.to_csv(RESULTS / "tau_choice.csv", index=False)
    lines = ["| Model | " + " | ".join(f"R = {r}" for r in RATIOS) + " |",
             "|---|" + "---|" * len(RATIOS)]
    for a in ALGOS[1:]:
        cells = []
        for r in RATIOS:
            s = sens[(sens["algo"] == a) & (sens["R"] == r)].iloc[0]
            cells.append(f"{s['falls_detected']}/30, {s['false_alerts_adl']} FA "
                         f"(tau {min(s['tau_per_fold']):.2f}-{max(s['tau_per_fold']):.2f})")
        lines.append(f"| {LABELS[a]} | " + " | ".join(cells) + " |")
    return "\n".join(lines), dict(zip(chosen["algo"], chosen["tau_final"]))


# --------------------------------------------------------------------------- 4. errors ----
def errors(seqs):
    rows = []
    kind = dict(zip(seqs["seq"], seqs["kind"]))
    for a in ALGOS[1:]:
        oof = load_oof(a)
        for s, g in oof.groupby("seq"):
            e = event_metrics(g, seqs)
            if kind[s] == "adl" and e["false_alerts_adl"]:
                first = g.loc[g["p1"] >= 0.5].iloc[0]
                rows.append({"algo": a, "seq": s, "error": "false alert",
                             "count": e["false_alerts_adl"], "end_frame": int(first["end_frame"]),
                             "p1": round(first["p1"], 2)})
            if kind[s] == "fall" and e["n_detected"] == 0:
                rows.append({"algo": a, "seq": s, "error": "missed fall", "count": 1,
                             "end_frame": int(seqs.set_index("seq").loc[s, "fall_start"]) + 30,
                             "p1": round(g["p1"].max(), 2)})
    err = pd.DataFrame(rows)
    err.to_csv(RESULTS / "errors.csv", index=False)
    by_seq = err.groupby(["error", "seq"])["algo"].apply(lambda x: ", ".join(x)).reset_index()
    by_seq["n_models"] = by_seq["algo"].str.count(",") + 1
    return by_seq.sort_values(["error", "n_models"], ascending=[True, False])


# ---------------------------------------------------------------- 5. external test ----
def external(tau_final, name="gmdcsa24"):
    p = PROCESSED / name
    meta = pd.read_parquet(p / "windows_meta.parquet")
    feats = pd.read_parquet(p / "features.parquet")
    X = np.load(p / "X_seq.npy")
    seqs = pd.read_csv(p / "sequences.csv")
    rows = []
    for a in ALGOS:
        proba = load_predict_proba(a)(feats, X)
        oof = pd.DataFrame({"window_id": meta["window_id"], "seq": meta["seq"], "fold": 0,
                            "end_frame": meta["end_frame"], "y_true": meta["y"],
                            "p0": proba[:, 0], "p1": proba[:, 1], "p2": proba[:, 2],
                            "y_pred": proba.argmax(axis=1)})
        for tau_name, t in [("0.5", 0.5), ("chosen", tau_final.get(a, 0.5))]:
            m = evaluate(oof, seqs, t)
            rows.append({"algo": a, "tau": tau_name, "tau_value": t, "pr_auc_fall": m["pr_auc_fall"],
                         "f1_fall": m["f1_fall"], "f1_lying": m["f1_lying"],
                         "falls_detected": m["n_detected"], "n_falls": m["n_falls"],
                         "false_alerts_adl": m["false_alerts_adl"],
                         "adl_seqs_with_false_alert": m["adl_seqs_with_false_alert"],
                         "false_alerts_per_hour": m["false_alerts_per_hour"]})
    ext = pd.DataFrame(rows)
    ext.to_csv(RESULTS / f"external_{name}.csv", index=False)
    n_adl = (seqs["kind"] == "adl").sum()
    lines = ["| Model | PR-AUC fall | F1 lying | tau 0.5: falls detected | tau 0.5: false alerts "
             f"(ADL clips alerted / {n_adl}) | chosen tau | chosen tau: falls detected | "
             "chosen tau: false alerts |", "|---|---|---|---|---|---|---|---|"]
    for a in ALGOS:
        d, c = ext[(ext.algo == a) & (ext.tau == "0.5")].iloc[0], ext[(ext.algo == a) & (ext.tau == "chosen")].iloc[0]
        lines.append(f"| {LABELS[a]} | {d.pr_auc_fall:.2f} | {d.f1_lying:.2f} | "
                     f"{d.falls_detected}/{d.n_falls} | {d.false_alerts_adl} ({d.adl_seqs_with_false_alert}) | "
                     f"{c.tau_value:.2f} | {c.falls_detected}/{c.n_falls} | {c.false_alerts_adl} |")
    return "\n".join(lines)


def main():
    seqs = pd.read_csv(PROCESSED / "sequences.csv")
    table = comparison(seqs)
    figures(seqs)
    tau_md, tau_final = tau_choice(seqs)
    err = errors(seqs)
    ext_md = {n: external(tau_final, n) for n in EXTERNAL}
    err_md = "| Error | Sequence | Models | n |\n|---|---|---|---|\n" + "\n".join(
        f"| {r.error} | {r.seq} | {r.algo} | {r.n_models} |" for r in err.itertuples())
    text = (f"# Comparison (generated by src/compare.py)\n\n## URFD, out-of-fold, tau = 0.5\n\n"
            f"{table}\n\nFold mean ± std over the 5 folds; counts pooled over the folds.\n\n"
            f"## Alert threshold chosen without the test fold (R = cost missed fall / cost false "
            f"alert; R = {R_IMPACT} from the impact model, other values for sensitivity)\n\n{tau_md}\n\nFinal tau (all URFD OOF, R = {R_DEFAULT}): "
            f"{tau_final}\n\n## Errors at tau = 0.5 (sequence, models making it)\n\n{err_md}\n\n"
            + "".join(f"## External test: {EXTERNAL[n]} (final models, never tuned on it)\n\n{md}\n\n"
                      for n, md in ext_md.items()))
    (RESULTS / "comparison.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
