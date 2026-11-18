# P5 — Final comparison

Everything below is recomputed by `src/compare.py` with `src/metrics.py` from the six
`results/<algo>_oof.parquet` (same windows, labels and 5 folds for all). Full tables:
`results/comparison.md`. Figures: `pr_curves.png`, `confusion_matrices.png`, `tau_tradeoff.png`,
`errors/*.png`. Rerun: `python src/compare.py && python src/confirm.py && python src/error_images.py`.

## 1. URFD, out-of-fold, tau = 0.5

| Model | PR-AUC fall | Falls detected | Delay (s) | False alerts ADL (raw) | Latency (ms) | Size |
|---|---|---|---|---|---|---|
| Naive | 0.07 ± 0.01 | 0 / 30 | — | 0 | 0.2 | 1 KB |
| Threshold rules | 0.12 ± 0.04 | 28 / 30 | 0.60 | 30 | 0.2 | < 1 KB |
| Logistic regression | 0.62 ± 0.12 | 29 / 30 | 0.35 | 37 | 0.7 | 4 KB |
| Decision tree | 0.33 ± 0.09 | 29 / 30 | 0.37 | 35 | 0.6 | 7 KB |
| **Gradient boosting** | **0.67 ± 0.10** | **29 / 30** | **0.37** | **15** | 1.0 | 1.1 MB |
| Neural network | 0.60 ± 0.12 | 25 / 30 | 0.38 | 24 | 0.1 | 561 KB |

The 40 ADL sequences give only ~4 minutes of monitoring: 15 false alerts is "~200 / h" on paper,
but that rate is very imprecise and over-estimates daily life (URFD ADLs are dense in risky
movements). All models are fast enough for real time (≤ 1 ms per window).

- Boosting is best on every alert metric. Logistic regression and the network rank windows almost
  as well (PR-AUC within one fold std), CART clearly worse, rules cannot rank (0/1 output).
- `lying` is easy for every learned model (F1 0.86-0.90). The hard part is `fall` vs a fast
  `not fall` movement.

## 2. Alert threshold (tau) without the test fold

For each fold, tau minimises `R × missed falls + false alerts` on the 4 other folds' predictions,
then the fold is scored with it. `R` (cost of a missed fall / cost of a false alert) is a
**business hypothesis**, not a sourced number (`config/impact.yaml` is still to come):

| Model | R = 1 | R = 3 | R = 10 | R = 30 |
|---|---|---|---|---|
| Logistic regression | 24/30, 8 FA | 28/30, 12 FA | 28/30, 16 FA | 28/30, 23 FA |
| Decision tree | 28/30, 30 FA | 29/30, 31 FA | 29/30, 31 FA | 29/30, 43 FA |
| Gradient boosting | 23/30, 6 FA | 27/30, 15 FA | 27/30, 15 FA | 27/30, 15 FA |
| Neural network | 20/30, 6 FA | 20/30, 11 FA | 25/30, 31 FA | 28/30, 34 FA |

- With a tuned tau (0.85-0.90), **logistic regression matches the boosting** (28/30 and 16 false
  alerts vs 27/30 and 15): its weakness at tau = 0.5 was the threshold, not the model.
- The network is the least stable: its best tau jumps from 0.1 to 0.95 with R.
- The choice of tau is a business decision: it should be set with the BSPP from the real cost of
  a useless dispatch vs an hour on the floor (`tau_tradeoff.png` shows the whole trade-off).

## 3. Errors (`results/errors.csv`, `results/errors/`)

- **False alerts:** 5 ADL sequences fool all 5 models (adl-05, 06, 16, 33, 37). The ones we looked
  at are a person **bending down quickly to pick something up** (`errors/boosting_adl-05.png`,
  `adl-04`, `adl-10`): within one second of skeleton, the torso tilts and the hips drop like a fall.
  URFD labels these frames "not lying", so they count as `not fall`. A quarter to a third of the
  false alerts are followed by the person lying down on purpose (bed, floor).
- **Missed falls:** fall-03 is missed by 4 models (`errors/boosting_fall-03.png`): a backward fall
  very close to the camera, the body is cut by the frame at the start, then collapses into a
  sitting position. The network misses 4 more falls that the other models catch.
- **Alert confirmation by lying (P3, `src/confirm.py`, rerun on every model):** alert only if
  the person is then seen lying. Boosting goes from 29/30 and 15 false alerts to **27/30 and 7**,
  for +1 s of delay (1.4 s). It cuts the false alerts of every model by 40-55 %; the price is 1-8 missed
  falls, mostly because YOLO misses ~35 % of lying frames.

## 4. External test: GMDCSA-24 (other room, camera and people; never tuned on it)

| Model | PR-AUC fall URFD → GMDCSA | Falls detected (tau 0.5) | False alerts (ADL clips alerted / 81) |
|---|---|---|---|
| Threshold rules | 0.12 → 0.15 | 67 / 78 | 66 (53) |
| Logistic regression | 0.62 → **0.45** | 63 / 78 | 60 (47) |
| Decision tree | 0.33 → 0.22 | 66 / 78 | 77 (52) |
| Gradient boosting | 0.67 → 0.42 | 53 / 78 | **33 (28)** |
| Neural network | 0.60 → **0.18** | 54 / 78 | 63 (50) |

- **Performance drops for every model** when the room, camera and people change: fall recall
  from ~97 % to 68-86 %, and every model but the boosting alerts on more than half of the ADL
  clips.
- The **network collapses** (PR-AUC 0.60 → 0.18): on raw skeletons it learned URFD's camera
  angle. The hand-made features (torso angle, hip speed, head height, all normalised by the torso
  length) transfer much better.
- Boosting keeps the fewest false alerts (33, 28 clips out of 81); logistic regression keeps the
  best ranking (0.45). With the tau chosen on URFD (0.85), logistic regression goes to 53/78 falls
  and 26 false alerts, close to the boosting.
- GMDCSA-24 labels are intervals: the fall class is approximated as the first second of the
  "Falling" interval (see `src/external.py`). The event metrics (fall start) do not depend on it.

## 5. Conclusion

**Retained model: gradient boosting (balanced) + alert confirmation by lying.** It has the best
URFD results at the default threshold (29/30 falls, 15 false alerts; 27/30 and 7 with confirmation)
and the fewest false alerts on the external set. **Runner-up: logistic regression** — same level
once its threshold is tuned, 300× smaller, readable coefficients, and the best ranking on
GMDCSA-24. If the decision-maker values explainability, it is a defensible choice.

What we **cannot** claim:
- A false-alert rate for real life: 4 minutes of ADL, and still far too many alerts per hour for
  a dispatch service as is. A deployed system would need confirmation (lying, no movement, a call
  to the person) before sending anyone.
- Robustness across homes: one camera set-up for training, 70 sequences, and a clear drop on a
  second dataset.
- The cost ratio R: it is a hypothesis until `config/impact.yaml` gives sourced values.

**Still to do:** the ultimate test on the team's own videos (`data/own/` is empty: put the
videos and `labels.csv` there, then `python src/external.py own` and `python src/demo.py <video>`),
`config/impact.yaml`, slides.
