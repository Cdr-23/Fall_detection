# P3 — Decision tree and gradient boosting

Inputs: `processed/features.parquet` (26 features), 5 outer folds by sequence (contract 7.0).
Hyper-parameters chosen by an inner 4-fold split **by sequence** inside the training folds
(`StratifiedGroupKFold` on `kind`), score = PR-AUC of class `fall`. Never tuned on the test fold.
Both models balance the classes (CART: `class_weight='balanced'`; boosting: balanced sample
weights), which shifts `p1` upward; `tau` is to be swept by P5. A third model,
`boosting_unweighted`, is the same boosting without class balancing (plain probabilities).

Run: `python src/run_cart.py` (~10 s), `python src/run_boosting.py` (~5 min, 8 cores),
`python src/run_boosting_unweighted.py` (~5 min), `python src/run_boosting.py --shap-only`
(SHAP only, from the saved model), then `python src/evaluate.py results/<algo>_oof.parquet`;
`python src/confirm.py` (alert confirmation, all `results/*_oof.parquet`).

## Results (URFD, out-of-fold, tau = 0.5; mean ± std over the 5 folds)

| | naive | CART | Boosting (balanced) | Boosting (unweighted) |
|---|---|---|---|---|
| ROC-AUC `fall` | 0.50 | 0.83 ± 0.06 | 0.91 ± 0.08 | 0.91 ± 0.08 |
| PR-AUC `fall` | 0.075 | 0.33 ± 0.10 | 0.67 ± 0.10 | 0.67 ± 0.11 |
| F1 `fall` | 0 | 0.47 ± 0.13 | 0.65 ± 0.08 | 0.61 ± 0.12 |
| F1 `lying` | 0 | 0.86 ± 0.02 | 0.90 ± 0.01 | 0.88 ± 0.01 |
| Falls detected (events) | 0 / 30 | 29 / 30 | 29 / 30 | 26 / 30 |
| Mean delay (s) | — | 0.37 | 0.37 | 0.41 |
| False alerts in ADL (raw count) | 0 | 35 | 15 | 8 |
| Latency per window (ms) | — | 0.8 | 1.2 | 1.2 |
| Model size (KB) | — | 7 | 1 128 | 1 119 |

- **The two boosting variants rank windows equally well** (same PR-AUC, the difference is far
  below the fold std); the unweighted one simply sits at a different point of the same
  precision/recall curve at `tau = 0.5` (fewer alerts, fewer falls detected). Its `p1` is a
  plain probability (mean 0.062 vs true rate 0.075; balanced: 0.085), better suited to a
  cost-based choice of `tau` by P5.

- Chosen hyper-parameters: `results/cart_params.csv`, `results/boosting_params.csv`. CART final:
  `max_depth=6, min_samples_leaf=10`; boosting final: `300 trees, max_depth=3, lr=0.1`.
- **Overfitting:** the boosting reaches PR-AUC 1.00 on its own training data versus 0.64
  out-of-fold; CART 0.52 versus 0.32. Only the out-of-fold numbers are meaningful.
- No leak suspected: ROC-AUC 0.91 is high but not near 1, and folds are by sequence.
- False alerts per hour are not meaningful at this scale (0.07 h of monitored ADL): give the
  raw count. Both models are at `tau = 0.5` with balanced weights; a higher `tau` will trade
  recall for fewer false alerts (P5).

## Alert confirmation by lying (proposal for P5 and the slides; `src/confirm.py`)

**Problem.** At `tau = 0.5` the boosting's false alerts are confident (median `p1` = 0.86):
fast sitting or bending looks like a fall within one second of skeleton, so raising `tau` alone
removes few of them without losing falls. What separates a fall is what comes next: the
person stays on the floor.

**Rule** (post-processing on any model's `p0/p1/p2`, same for every algorithm): a window with
`p1 >= 0.5` is only a *candidate*; the alert is raised when, within the next **3 s** (90
frames), the model predicts `lying` (`p2 >= 0.5`) on **3 consecutive windows** (~0.5 s).
Parameters fixed a priori, not tuned; `results/confirm_sensitivity.csv` varies them (horizon
2 / 3 / 5 s, 1 / 3 / 6 windows) for information only. Scored with the shared event metrics,
tolerance extended by the 3 s horizon (the alert is later by construction).

| (URFD, out-of-fold) | Falls detected | Mean delay | False alerts ADL (raw) |
|---|---|---|---|
| CART, raw alert | 29 / 30 | 0.37 s | 35 |
| CART, confirmed | 23 / 30 | 1.49 s | 16 |
| Boosting, raw alert | 29 / 30 | 0.37 s | 15 |
| Boosting, confirmed | 27 / 30 | 1.40 s | 7 |
| Boosting unweighted, raw alert | 26 / 30 | 0.41 s | 8 |
| Boosting unweighted, confirmed | 26 / 30 | 1.36 s | **3** |

Caution: these combinations were not chosen in advance against each other; picking the "best"
row from this table would be choosing on the test folds. Report them side by side, and let P5
choose the operating point (`tau`, confirmation or not) on validation folds and on cost.

- **False alerts halved** for ~1 s of extra delay, negligible next to a time on the floor
  counted in minutes or hours: this is the decision-relevant trade-off for a telecare operator.
- **The 2 falls lost by the boosting are a URFD artefact:** `fall-13` and `fall-21` end less
  than 1.5 s after the fall, so there is no time to see the person on the floor. A real camera
  keeps filming. (`fall-03` is missed in both cases: `p1` never exceeds 0.02.)
- **The 7 remaining false alerts all end in a voluntary lying down** (`y_true = lying` in the
  ADL: bed, sofa, floor). The rule cannot separate "lying down fast" from "falling then lying";
  that would need a longer context or the place (bed vs floor).
- **Team decision needed:** this changes the alert definition (and the delay tolerance of the
  event metrics). If adopted, apply it to every model in P5, not only to P3's.

## Decision tree (slides: `results/cart_tree_top.png`, full: `cart_tree.png`, `cart_rules.txt`)

The first splits read like rules: a **flat box** (`bbox_ratio_mean <= 1.08`, wider than tall)
means `lying`; otherwise a **torso that rotates during the window** (`torso_angle_delta > 8°`)
with the person well detected means `fall`. This mirrors P2's threshold rules (torso angle,
vertical motion), but the tree picks the bounding-box ratio first.

## SHAP (boosting, class `fall`: `results/shap_summary.png`, `results/shap_importance.csv`)

`shap.TreeExplainer` does not support a multiclass `GradientBoostingClassifier`; we explain each
regression tree of the class and sum (exact: SHAP is additive; checked against
`decision_function`, error 2e-14).

1. The most important feature is **`torso_angle_delta`**: the torso tilting from vertical
   toward horizontal within the second pushes toward `fall`. This is the physics of a fall.
2. Then **the head**: a head dropping relative to the hips (`head_height_delta` negative) pushes
   toward `fall`, while a head still above the hips at some point of the window
   (`head_height_min` high) separates `fall` (in progress) from `lying` (already down).
   Hips going down throughout the window (`hip_vy_min` high) also push toward `fall`.
3. Less intuitive: **low** `hip_ay_std` (smooth acceleration) pushes toward `fall`, and high
   values toward `not fall`. It may reflect jittery keypoints in some ADL rather than a physical
   effect; with 30 falls from one camera, this should not be over-interpreted.
