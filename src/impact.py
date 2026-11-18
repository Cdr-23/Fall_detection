"""Impact model: what the retained model is worth for a telecare operator (CLAUDE.md section 11).

Parameters: config/impact.yaml (each one sourced or marked as a hypothesis, with a low/high range).
For N people monitored over one year:
  long lies             = N x falls/person/year x long-lie share
  long lies avoided     = long lies x camera coverage x detection rate x (1 - share already
                          avoided by the pendant alarm)
  value                 = long lies avoided x cost of a hospitalized fall x avoidable share
  false-alert cost      = N x 365 x false alerts/person/day x (check call + dispatch share x
                          dispatch cost)
  net value             = value - false-alert cost - N x system cost
Also: break-even false-alert rate, cost ratio R = cost of a missed fall / cost of a false alert
(used to choose the alert threshold in src/compare.py), and a one-at-a-time sensitivity analysis.

Outputs: results/impact.md, results/impact_scenarios.csv, results/impact_sensitivity.csv,
results/impact_tornado.png,
results/impact_break_even.png.
Usage: python src/impact.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "impact.yaml").read_text())
RESULTS = ROOT / "results"
PARAMS = [k for k, v in CFG.items() if isinstance(v, dict) and "value" in v]
SCENARIO_KEYS = ["false_alerts_per_person_day", "avoidable_share_of_cost",
                 "dispatch_share_of_false_alerts", "detection_rate"]


def every(rate):
    """0.05 false alerts per day -> 'one every 20 days'."""
    return "none" if rate <= 0 else f"one every {1 / rate:.0f} days"


def compute(p, n=CFG["population"]):
    long_lies = n * p["falls_per_person_year"] * p["long_lie_share"]
    avoided = long_lies * p["camera_coverage"] * p["detection_rate"] * (1 - p["alarm_used_share"])
    value = avoided * p["cost_hospitalized_fall_eur"] * p["avoidable_share_of_cost"]
    cost_fa = p["cost_check_call_eur"] + p["dispatch_share_of_false_alerts"] * p["cost_dispatch_eur"]
    n_fa = n * 365 * p["false_alerts_per_person_day"]
    fa_cost = n_fa * cost_fa
    system = n * p["system_cost_per_person_year_eur"]
    net = value - fa_cost - system
    break_even = (value - system) / (n * 365 * cost_fa)
    missed_cost = p["long_lie_share"] * p["cost_hospitalized_fall_eur"] * p["avoidable_share_of_cost"]
    return {"long_lies": long_lies, "long_lies_avoided": avoided, "value_eur": value,
            "false_alerts": n_fa, "cost_per_false_alert_eur": cost_fa,
            "false_alert_cost_eur": fa_cost, "system_cost_eur": system, "net_eur": net,
            "net_per_person_eur": net / n, "break_even_false_alerts_per_day": break_even,
            "R_missed_fall_vs_false_alert": missed_cost / cost_fa}


def main():
    base = {k: CFG[k]["value"] for k in PARAMS}
    r = compute(base)
    rows = []
    for k in PARAMS:
        lo, hi = CFG[k]["low"], CFG[k]["high"]
        if lo == hi:
            continue
        a, b = compute({**base, k: lo})["net_eur"], compute({**base, k: hi})["net_eur"]
        rows.append({"parameter": k, "low": lo, "high": hi, "net_at_low_eur": round(a),
                     "net_at_high_eur": round(b), "swing_eur": round(abs(b - a)),
                     "kind": "hypothesis" if CFG[k].get("hypothesis") else "sourced/measured"})
    sens = pd.DataFrame(rows).sort_values("swing_eur", ascending=False)
    sens.to_csv(RESULTS / "impact_sensitivity.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, s in enumerate(sens.iloc[::-1].itertuples()):
        lo, hi = sorted([s.net_at_low_eur, s.net_at_high_eur])
        ax.barh(i, (hi - lo) / 1e3, left=lo / 1e3, color="#c44" if s.kind == "hypothesis" else "#47a")
    ax.axvline(r["net_eur"] / 1e3, color="k", lw=1)
    ax.axvline(0, color="grey", ls=":")
    ax.set_yticks(range(len(sens)), sens["parameter"].iloc[::-1], fontsize=8)
    ax.set_xlabel(f"Net value per {CFG['population']} people per year (thousand EUR); red = hypothesis")
    ax.set_title("Sensitivity of the net value (one parameter at a time)")
    fig.tight_layout()
    fig.savefig(RESULTS / "impact_tornado.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    rates = [i / 1000 for i in range(0, 301, 5)]
    ax.plot(rates, [compute({**base, "false_alerts_per_person_day": x})["net_per_person_eur"] for x in rates],
            color="#47a", label="base values")
    for k, lab in [("high", "favourable avoidable share (0.4)"), ("low", "unfavourable avoidable share (0.05)")]:
        q = {**base, "avoidable_share_of_cost": CFG["avoidable_share_of_cost"][k]}
        ax.plot(rates, [compute({**q, "false_alerts_per_person_day": x})["net_per_person_eur"] for x in rates],
                ls="--", lw=1, label=lab)
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(r["break_even_false_alerts_per_day"], color="k", ls=":", lw=1)
    ax.set(xlabel="False alerts per person per day (real-life rate, unknown)",
           ylabel="Net value per person per year (EUR)",
           title=f"Break-even at {r['break_even_false_alerts_per_day']:.2f} false alerts / person / day")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "impact_break_even.png", dpi=150)

    scen = {name: compute({**base, **over}) | {**base, **over}
            for name, over in CFG["scenarios"].items()}
    pd.DataFrame(scen).T.to_csv(RESULTS / "impact_scenarios.csv", index_label="scenario")
    be = r["break_even_false_alerts_per_day"]
    lines = [f"# Impact model (generated by src/impact.py from config/impact.yaml)\n",
             f"For {CFG['population']} people living alone with telecare, monitored for one year. "
             "Every parameter is sourced or marked as a hypothesis in `config/impact.yaml`.\n",
             "## When does it pay off?\n",
             f"The decisive unknown is the **real-life false-alert rate**, which public videos cannot "
             f"measure. With the central values, the system pays off if false alerts stay below "
             f"**{be:.3f} per person per day ({every(be)})**.\n",
             "| Scenario | False alerts per person | Avoidable share of a hospitalized fall's cost | "
             "Detection rate | Long lies avoided | Net value per person per year |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {name} | {every(s['false_alerts_per_person_day'])} | {s['avoidable_share_of_cost']:.0%} | "
              f"{s['detection_rate']:.0%} | {s['long_lies_avoided']:.0f} | **{s['net_per_person_eur']:+,.0f} EUR** |"
              for name, s in scen.items()]
    lines += ["", "## Health benefits not counted in euros\n",
              f"- About **{r['long_lies_avoided']:.0f} long lies (≥ 1 h on the floor) avoided per year** "
              f"(central; {scen['pessimistic']['long_lies_avoided']:.0f} to "
              f"{scen['optimistic']['long_lies_avoided']:.0f} across scenarios), out of "
              f"{r['long_lies']:.0f}."]
    lines += [f"- {c}" for c in CFG["health_context"]]
    lines += ["- The euros above only count part of the hospital cost; deaths, moves into care homes "
              "and quality of life are not monetized, so the value is a lower bound.",
              "", "## Central scenario in detail\n",
              "| Quantity | Value |", "|---|---|",
             f"| Long lies (≥ 1 h on the floor) | {r['long_lies']:.0f} |",
             f"| Long lies avoided by the camera | {r['long_lies_avoided']:.0f} |",
             f"| Value of avoided long lies | {r['value_eur']:,.0f} EUR |",
             f"| False alerts | {r['false_alerts']:,.0f} (cost {r['cost_per_false_alert_eur']:.2f} EUR each) |",
             f"| Cost of false alerts | {r['false_alert_cost_eur']:,.0f} EUR |",
             f"| System cost | {r['system_cost_eur']:,.0f} EUR |",
             f"| **Net value** | **{r['net_eur']:,.0f} EUR ({r['net_per_person_eur']:.0f} EUR / person / year)** |",
             f"| Break-even false-alert rate | {be:.3f} per person per day ({every(be)}) |",
             f"| R = cost of a missed fall / cost of a false alert | {r['R_missed_fall_vs_false_alert']:.0f} |",
             "", "## Sensitivity (one parameter at a time, low → high)", "",
             "| Parameter | Range | Net value (EUR) | Kind |", "|---|---|---|---|"]
    lines += [f"| {s.parameter} | {s.low} – {s.high} | {s.net_at_low_eur:,} → {s.net_at_high_eur:,} | {s.kind} |"
              for s in sens.itertuples()]
    (RESULTS / "impact.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("->", RESULTS / "impact.md", RESULTS / "impact_sensitivity.csv", RESULTS / "impact_tornado.png",
          RESULTS / "impact_break_even.png")


if __name__ == "__main__":
    main()
