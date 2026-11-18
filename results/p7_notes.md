# P7: floor-height and 3-second look-back features (experiment)

Goal: separate "falling" from "lying down on purpose", the cause of the 7 confirmed false alerts
left by the retained model on URFD. Two ideas and three variants were fixed in advance; nothing
(threshold, feature, parameter) was tuned on a test fold, GMDCSA-24 or Le2i. Same windows, labels,
folds, model, grid and inner tuning as `src/run_boosting.py`; only the columns change.
Code: `src/features_ctx.py`, `src/check_features_ctx.py`, `src/run_boosting_ctx.py`,
`src/eval_ctx.py`. Full tables: `results/p7_comparison.md`, `results/p7_feature_checks.md`.

## Features (20, all causal on the prepared poses)

- **FLOOR (6).** Floor reference = median y of the mid-ankles over the last 30 "standing" frames
  seen so far (detected, torso < 30°, both ankles), NaN before 10 such frames.
  `hip_above_floor` and `head_above_floor` = (floor − y) / torso length; per window `_mean`,
  `_min`, `_last`.
- **3 s look-back (6),** frames [end − 89, end]: `hip_drop_3s`, `descent_time_3s` (time since the
  highest hip position), peaks `hip_vy_max_3s`, `hip_ay_max_3s`, `torso_angle_min_3s`,
  `head_height_max_3s`.
- **Previous windows (8):** `torso_angle_mean`, `head_height_mean`, `bbox_ratio_mean`, `hip_vy_max`
  on the window ending 1 s (`_lag1s`) and 2 s (`_lag2s`) earlier; NaN if it starts before the clip.
- **Limits.** Perspective: lying closer to or farther from the camera than where the person stood
  shifts the floor reference, and values below the floor (< 0) are common. Lying down on purpose
  *on the floor* is not covered. The torso-length scale uses the whole sequence, as in the base
  pipeline. In URFD the look-back is mostly empty for falls: 16 of 30 falls start before frame 30.

## Checks (`results/p7_feature_checks.md`)

- **Hand recomputation.** One feature per group, recomputed with plain loops on 3 random
  sequences (fall-06, adl-07, fall-14): identical values. `_lag1s` also equals the stored base
  feature of the window ending 30 frames earlier.
- **Causality of the new code.** Overwriting every frame after the window end with noise changes
  nothing (3/3 sequences). End to end, the **shared** preprocessing interpolates gaps of up to
  0.5 s using the next valid frame. In fall-14 this changed 3 of 7 earlier windows, both in the
  base features and in the context ones. This look-ahead already exists in the retained pipeline.
- **Orders of magnitude** (`results/p7_floor_check.png`, median `hip_above_floor_mean`): upright
  1.38, lying after a fall 0.50, **lying on purpose 0.18**. In URFD, voluntary lying is not
  higher above the floor than lying after a fall, so the floor idea has no signal to learn here.
- **NaN.** The `_lag` features are NaN for 61 % (1 s) and 83 % (2 s) of URFD fall windows, against
  22 % and 42 % of not-fall windows (Le2i: 0 %). The FLOOR features are NaN for 2-15 % of windows.

## Results (alert p1 ≥ 0.5; "conf." = `confirm.confirm_alerts` defaults)

| Model | URFD PR-AUC | URFD falls / 30, raw → conf. | URFD FA, raw → conf. | GMDCSA-24 falls / 78, FA (conf.) | Le2i falls / 37, FA (conf.) |
|---|---|---|---|---|---|
| **boosting (retained)** | 0.67 ± 0.10 | 29 → 27 | 15 → 7 | 52, 9 | 22, 1 |
| boosting_floor | 0.66 ± 0.17 | 29 → 24 | 19 → 9 | 43, 11 | 24, 1 |
| boosting_lookback | 0.73 ± 0.14 | 27 → 24 | 14 → 6 | 58, 17 | 14, 5 |
| boosting_ctx (candidate) | 0.75 ± 0.12 | 27 → 24 | 11 → 6 | 51, 14 | 12, 2 |

External PR-AUC: GMDCSA-24 0.42 (boosting) vs 0.35 (ctx); Le2i 0.30 vs 0.09. Latency is ≤ 1.4 ms
per window for every variant. Note: CLAUDE.md gives 2 confirmed Le2i false alerts for boosting
because it comes from `confirm_long.py` (D = 0.5 s, horizon 105 frames); with the `confirm.py`
defaults used here (90 frames) the count is 1. URFD and GMDCSA-24 are identical either way.

**The 7 confirmed URFD false alerts.** boosting_ctx removes 3 (adl-10, adl-21, adl-31) but adds 2
(adl-36, adl-40), for a net change of 7 → 6. It also loses 3 more falls (fall-05, fall-07, fall-16),
going from 27 to 24 detected.

**Leak check.** Le2i is the only set where every fall has a full 3 s history, and both look-back
variants collapse there (raw 17 → 7-8 falls, PR-AUC 0.30 → 0.09). ADL windows with no history
(ending before frame 60) are alerted more often by boosting_lookback (4.3 %) than later ones
(2.5 %); the retained boosting shows the opposite (1.3 % vs 2.8 %). In SHAP (class fall), the
`_lag` features carry the same share for early and late falls (26 % vs 24 %): median imputation
hides NaN as such. Look-back features rank 3rd, 5th and 6th overall (`results/p7_shap_summary.png`).
These signs fit the feared artefact: the model learns how URFD clips are cut (the fall comes right
after the clip starts), not how a fall happens. We cannot separate this from a plain camera change.

## Reading and recommendation

- On URFD, the gains (PR-AUC +0.08, 1 false alert fewer after confirmation) are smaller than the
  fold standard deviation (0.10-0.12). With 30 falls and 7 false alerts this is not a result.
  After confirmation, boosting_ctx also detects 3 fewer falls.
- **The external tests do not improve; they get worse:** more false alerts on GMDCSA-24 (9 → 14)
  and half the falls on Le2i (22 → 12). The floor variant alone is mixed: better on Le2i (24 vs 22),
  worse on URFD (24 vs 27) and on GMDCSA-24 (43 vs 52).
- **Recommendation: keep the retained model (boosting + confirmation). Do not propose
  boosting_ctx.** The look-back idea cannot be tested on URFD, whose clips start just before the
  fall. Testing it needs training data with several seconds of activity before each fall (Le2i,
  the team videos), with a held-out dataset. The real fix for voluntary lying is still the long
  confirmation (10-30 s on the floor) plus the check call.
