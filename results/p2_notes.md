# P2 — Threshold rules and logistic regression

Ported from Alain's branch `p2-rules` (`src/p2.py`) to the shared pipeline: same windows, labels
and 5 outer folds as P1/P3/P4 (`common.load_data()`, contract 7.0). Alain's own preparation
(`prepare.py`) counted URFD label 0 as `fall` in ADL sequences too (314 fall windows instead of 151)
and used different folds, so its scores were not comparable. The rule logic and the logistic
regression set-up are his.

Run: `python src/run_rules.py`, `python src/run_logreg.py`, then
`python src/evaluate.py results/<algo>_oof.parquet`.

## Models

- **Rules** (`src/run_rules.py`): `lying` if `torso_angle_mean ≥ t_lie` and
  `head_height_mean ≤ h_lie`; else `fall` if `hip_vy_max ≥ v_fall` and `torso_angle_max ≥ t_fall`;
  else `not fall`. `lying` is checked first: lying windows have even larger hip speeds than falls
  (keypoint jitter on a body on the floor), so the fall rule alone fires on them. Thresholds chosen
  by grid search on the training folds only (score = mean of F2 `fall` and F1 `lying`); final
  thresholds: `v_fall` 0.5, `t_fall` 30°, `t_lie` 20°, `h_lie` 0.5 (`results/rules_thresholds.csv`).
- **Logistic regression** (`src/run_logreg.py`): median imputer → scaler → multinomial logistic
  regression, `class_weight='balanced'`, 26 features. `C` chosen by inner split by sequence
  (PR-AUC `fall`, same helper as P3): 1 or 10 per fold, 1 for the final model.

## Results (URFD, out-of-fold, tau = 0.5; fold mean ± std)

| | naive | rules | **logreg** | CART | boosting | NN |
|---|---|---|---|---|---|---|
| ROC-AUC `fall` | 0.50 | 0.63 ± 0.07 | **0.91 ± 0.04** | 0.83 ± 0.06 | 0.91 ± 0.08 | 0.88 ± 0.08 |
| PR-AUC `fall` | 0.07 | 0.12 ± 0.04 | **0.62 ± 0.12** | 0.33 ± 0.09 | 0.67 ± 0.10 | 0.60 ± 0.12 |
| F1 `fall` | 0 | 0.24 ± 0.08 | **0.54 ± 0.03** | 0.47 ± 0.13 | 0.65 ± 0.08 | 0.48 ± 0.13 |
| F1 `lying` | 0 | 0.57 ± 0.15 | **0.87 ± 0.06** | 0.86 ± 0.02 | 0.90 ± 0.01 | 0.88 ± 0.05 |
| Falls detected | 0 / 30 | 28 / 30 | **29 / 30** | 29 / 30 | 29 / 30 | 25 / 30 |
| Mean delay (s) | — | 0.60 | **0.35** | 0.37 | 0.37 | 0.38 |
| False alerts in ADL (raw) | 0 | 30 | **37** | 35 | 15 | 24 |
| Model size (KB) | 1 | 0.07 | **3.7** | 7 | 1 128 | 561 |

All columns from `results/<algo>_metrics.csv` (same `evaluate.py`).

**Reading:** the rules catch most falls (28 / 30) but cannot rank windows (0/1 output, PR-AUC
0.12) and recognise `lying` poorly: many lying windows have no detected person (YOLO finds a person
on only 65 % of lying frames), so their features are NaN and the rules say `not fall`. The logistic
regression, with 26 features and a 4 KB model, is as good as the network on ranking (PR-AUC 0.62 vs
0.60) and close to the boosting (0.67), but at `tau = 0.5` it raises the most false alerts (37):
its threshold needs tuning in P5.

## Logistic regression coefficients (`results/logreg_coefficients.csv`)

Standardised, class `fall`, final model. Top: `hip_ay_std` −3.4, `bbox_ratio_mean` +2.3,
`valid_frac` +1.6, `head_height_std` −1.4, `hip_ay_max` +1.3, `hip_vy_max` +1.2.
- A fast downward movement (`hip_vy_max`, `hip_ay_max`) and a box still tall on average
  (`bbox_ratio_mean`: the window starts standing) push toward `fall`: it is the start of a fall.
- A very jittery acceleration (`hip_ay_std`) pushes away from `fall` and toward `lying`: it is
  keypoint noise on a body already on the floor, not a fall.
- The features are strongly correlated (mean / max / std of the same signal), so single
  coefficients should not be over-read; the boosting SHAP ranking (P3) is the safer reading.
