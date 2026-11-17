# P3 — Decision tree and gradient boosting

Inputs: `processed/features.parquet` (26 features), 5 outer folds by sequence (contract 7.0).
Hyper-parameters chosen by an inner 4-fold split **by sequence** inside the training folds
(`StratifiedGroupKFold` on `kind`), score = PR-AUC of class `fall`. Never tuned on the test fold.
Both models balance the classes (CART: `class_weight='balanced'`; boosting: balanced sample
weights), so `p1` overstates the real fall probability: `tau` is to be swept by P5.

Run: `python src/run_cart.py` (~10 s), `python src/run_boosting.py` (~5 min, 8 cores),
`python src/run_boosting.py --shap-only` (SHAP only, from the saved model), then
`python src/evaluate.py results/<algo>_oof.parquet`.

## Results (URFD, out-of-fold, tau = 0.5; mean ± std over the 5 folds)

| | naive | CART | Gradient boosting |
|---|---|---|---|
| ROC-AUC `fall` | 0.50 | 0.83 ± 0.06 | 0.91 ± 0.08 |
| PR-AUC `fall` | 0.075 | 0.33 ± 0.10 | 0.67 ± 0.10 |
| F1 `fall` | 0 | 0.47 ± 0.13 | 0.65 ± 0.08 |
| F1 `lying` | 0 | 0.86 ± 0.02 | 0.90 ± 0.01 |
| Falls detected (events) | 0 / 30 | 29 / 30 | 29 / 30 |
| Mean delay (s) | — | 0.37 | 0.37 |
| False alerts in ADL (raw count) | 0 | 35 | 15 |
| Latency per window (ms) | — | 0.8 | 1.2 |
| Model size (KB) | — | 7 | 1 128 |

- Chosen hyper-parameters: `results/cart_params.csv`, `results/boosting_params.csv`. CART final:
  `max_depth=6, min_samples_leaf=10`; boosting final: `300 trees, max_depth=3, lr=0.1`.
- **Overfitting:** the boosting reaches PR-AUC 1.00 on its own training data versus 0.64
  out-of-fold; CART 0.52 versus 0.32. Only the out-of-fold numbers are meaningful.
- No leak suspected: ROC-AUC 0.91 is high but not near 1, and folds are by sequence.
- False alerts per hour are not meaningful at this scale (0.07 h of monitored ADL): give the
  raw count. Both models are at `tau = 0.5` with balanced weights; a higher `tau` will trade
  recall for fewer false alerts (P5).

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
