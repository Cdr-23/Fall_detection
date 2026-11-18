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

## 3 bis. Longer confirmation: the person must stay on the floor D seconds (`src/confirm_long.py`)

Alert only if, within 3 s of a fall candidate, the windows stay classified `lying` for D seconds.
Results in `results/confirm_long.csv` (strict variant; the "lenient" one, where a window with no
detected person also counts as on the floor, gains 0-3 falls for 0-3 more false alerts).

| GMDCSA-24 (final models) | raw alert | D = 0.5 s | D = 1 s | D = 2 s | D = 3 s |
|---|---|---|---|---|---|
| Boosting: false alerts (81 ADL clips) | 33 | 9 | 8 | 6 | 3 |
| Boosting: falls detected / testable | 53 / 78 | 52 / 77 | 49 / 76 | 40 / 67 | 15 / 34 |
| Logistic regression: false alerts | 60 | 15 | 13 | 6 | 4 |
| Logistic regression: falls detected / testable | 63 / 78 | 58 / 77 | 55 / 76 | 41 / 67 | 15 / 34 |

On URFD (out-of-fold), boosting false alerts go 15 → 7 (D = 1 s) → 5 (2 s) → 3 (3 s) → 0 (5 s).

- Requiring ~1 s on the floor **divides the false alerts by 4** (boosting 33 → 8, logistic
  regression 60 → 13 on GMDCSA-24) for a small loss of falls and ~2 s of delay. The false alerts
  that remain are mostly people lying down on purpose.
- **Beyond ~1 s the data cannot answer.** URFD stops filming 1.1-3.1 s after a fall starts and
  GMDCSA-24 1.4-9.1 s after, so few falls are "testable" (filmed ≥ 1 s + D after the start), and
  the drop of fall recall at D = 2-3 s mixes model errors with clips that end too early. ADL clips
  last ~8-9 s, so a long D also removes false alerts partly because the clip ends. **10-30 s
  cannot be tested at all**: it needs our own videos where the person stays on the floor.

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

## 4 bis. Second external test: Le2i FDD, Home_01 + Home_02 (never tuned on it)

60 clips from two other homes (37 falls, 23 ADL; 24 fps resampled to 30; Kaggle mirror of Le2i).
Only the two home scenes were used: Lecture_room and Office have no annotations. The raw AVIs
(uncompressed bitmaps) crash OpenCV here, so `src/le2i_convert.py` rewrites them to MP4 first.
Labels: the annotated fall start, then the same rule as GMDCSA-24 (1 s `fall`, then `lying`).

| Model | PR-AUC fall | Falls detected (tau 0.5) | False alerts (ADL clips alerted / 23) | + 1 s on the floor: falls / false alerts |
|---|---|---|---|---|
| Threshold rules | 0.12 | 21 / 37 | 17 (13) | — |
| **Logistic regression** | 0.25 | **31 / 37** | 31 (18) | **32 / 35, 2** |
| Decision tree | 0.12 | 24 / 37 | 36 (21) | — |
| Gradient boosting | 0.30 | **17 / 37** | 7 (7) | 19 / 35, 0 |
| Neural network | 0.15 | 23 / 37 | 29 (23) | — |

(`results/external_le2i.csv`, `results/confirm_long.csv`; with confirmation, falls are counted on
the 35 clips filmed long enough, and a fall counts as detected if the confirmed alert comes within
the extended delay.)

- The drop is even larger than on GMDCSA-24 (PR-AUC ≤ 0.30 for every model).
- **The boosting misses more than half of the falls** (17 / 37): its few false alerts come from
  being too cautious in a new room. Logistic regression keeps most falls (31 / 37).
- With 1 s on the floor, logistic regression gives **32 / 35 falls and 2 false alerts**.

## 4 ter. Impact (`src/impact.py`, `config/impact.yaml`, `results/impact.md`)

For 1,000 people living alone with telecare, over one year (base values; each parameter sourced
or marked as a hypothesis in `config/impact.yaml`):
- ~180 long lies (≥ 1 h on the floor; Fleming & Brayne, BMJ 2008), of which the camera would
  cut short ~58 (one camera per home, detection 0.675 measured on GMDCSA-24, pendant alarms
  already used in a minority of cases).
- Value: ~138 k EUR (11,796 EUR extra cost in the year after a hospitalized fall, Santé publique
  France 2023, × a hypothetical 20 % avoided by a fast alert).
- **Everything depends on the real-life false-alert rate.** Each false alert costs ~10 EUR (a
  check call + 5 % of dispatches at 141.80 EUR, SDIS 81). The system breaks even below **~0.02
  false alerts per person per day** (one every ~7 weeks); at 0.2 / day it loses ~660 EUR per
  person per year (`impact_break_even.png`, `impact_tornado.png`).
- Cost ratio R (missed fall vs false alert) ≈ 70: a missed fall is much more costly, which
  supports a low alert threshold **followed by a confirmation step**, not a high threshold.
- Reading: the model is useful only inside a two-step process (confirmation on the floor for
  10-30 s, then a check call before any dispatch). Measuring the daily-life false-alert rate on
  long team videos is the next step that decides whether it pays off.

## 5. Conclusion

**Retained model: gradient boosting (balanced) + alert confirmation by lying.** It has the best
URFD results at the default threshold (29/30 falls, 15 false alerts; 27/30 and 7 with confirmation)
and the fewest false alerts on both external sets. **Runner-up: logistic regression** — same level
once its threshold is tuned, 300× smaller, readable coefficients, and the best ranking on
GMDCSA-24. If the decision-maker values explainability, it is a defensible choice.

The model is chosen on the URFD cross-validation, as planned; the external sets (GMDCSA-24,
Le2i) are only used to measure how much performance drops in a new setting, not to pick the model.

**Confirmation delay: measured vs product setting.** The results above use ~0.5 s lying (3
windows), the longest delay our datasets can validate. Boosting, raw alert → confirmed:

| | URFD (CV) | GMDCSA-24 | Le2i |
|---|---|---|---|
| Falls detected | 29 / 30 → 27 / 30 | 53 / 78 → 52 / 77 | 17 / 37 → 22 / 37 |
| False alerts (ADL clips) | 15 → 7 | 33 → 9 | 7 → 2 |
| Mean delay | 0.4 s → 1.4 s | 0.3 s → 1.4 s | 0.2 s → 0.9 s |

(Le2i gains falls because a confirmed alert may come a little later than the raw one.) For the
deployed product we recommend a **longer delay, 10-30 s on the floor**, as a setting chosen with
the operator: after a real fall the person stays down for minutes, so the cost is a few seconds,
and most remaining false alerts (bending, sitting) do not last that long. This is **not measured**:
public clips end 2-4 s after the fall. It will be scored on the team videos (`src/confirm_long.py`
computes 10, 20 and 30 s as soon as `processed/own` exists).

What we **cannot** claim:
- A false-alert rate for real life: 4 minutes of ADL, and still far too many alerts per hour for
  a dispatch service as is. A deployed system would need confirmation (lying, no movement, a call
  to the person) before sending anyone.
- Robustness across homes: one camera set-up for training, 70 sequences, and a clear drop on the
  two external datasets. On Le2i (camera placed high), the boosting misses more than half of the
  falls (17 / 37, section 4 bis): training on more camera set-ups is the first improvement to make.
- The money: the impact model (section 4 ter) rests on hypotheses for the avoidable share of
  costs and for the daily-life false-alert rate, which decides whether the system pays off.

**Still to do:** the ultimate test on the team's own videos (`data/own/` is empty: put the
videos and `labels.csv` there, then `python src/external.py own`, `python src/confirm_long.py`
and `python src/demo.py <video>`; stay on the floor >= 30 s after each fall), README, slides,
2-minute video.
