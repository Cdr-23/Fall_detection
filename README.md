# Fall detection from 2D skeletons

Detect falls of older people living alone from a home camera, for a telecare operator or the
Paris fire brigade (BSPP). A video goes in; every second the system outputs `not fall`, `fall`
or `lying`, and raises an alert on `fall`. Only the 2D skeleton is kept (privacy, runs on CPU).
MIT Sloan 15.071 *The AI Edge*, Project 1 (predictive AI), November 2026.

## Approach

1. **Pose:** YOLOv8n-pose (pretrained on COCO, used as is) → 17 keypoints per frame.
2. **Windows:** 30 frames (1 s at 30 fps), stride 5; label = class of the last frame.
   URFD labels → classes: `fall` = the 30 "falling" frames of a fall sequence, `lying` = on
   the floor, `not fall` = everything else (including sitting and bending in daily-life clips).
3. **Inputs:** 26 hand-made features (torso angle, hip vertical speed / acceleration, head
   height, box ratio: mean / min / max / std / delta over the window), or the raw normalised
   skeleton for the neural network.
4. **Models, all on the same windows and folds:** naive baseline, threshold rules (business
   rule), logistic regression, decision tree, gradient boosting (+ SHAP), MLP / 1D-CNN.
5. **Validation:** 5-fold cross-validation **by video** (`StratifiedGroupKFold`, seed 42);
   hyper-parameters tuned on an inner split by video, never on the test fold. Two external
   test sets never used for training or tuning.
6. **Alert:** `p(fall) ≥ 0.5`, then **confirmation**: the person must then be seen lying
   (~0.5 s). Event metrics: a fall is detected if an alert comes within 1.5 s of its start;
   false alerts within 2 s count as one.

## Results (retained model: gradient boosting + confirmation)

| Boosting, raw alert → confirmed | URFD (5-fold CV) | GMDCSA-24 (external) | Le2i (external) |
|---|---|---|---|
| Falls detected | 29/30 → 27/30 | 53/78 → 52/77 | 17/37 → 22/37 |
| False alerts in daily-life clips | 15 → 7 | 33 → 9 | 7 → 2 |
| Mean alert delay | 0.4 s → 1.4 s | 0.3 s → 1.4 s | 0.2 s → 0.9 s |

URFD, PR-AUC of `fall` (fold mean ± std): boosting 0.67 ± 0.10, logistic regression
0.62 ± 0.12, neural network 0.60 ± 0.12, decision tree 0.33 ± 0.09, rules 0.12, naive 0.07.
All models run in ~1 ms per window on a laptop CPU. Full tables, threshold choice, error
analysis and limits: [`results/p5_notes.md`](results/p5_notes.md), [`results/comparison.md`](results/comparison.md).

**Impact** ([`config/impact.yaml`](config/impact.yaml) → [`results/impact.md`](results/impact.md); each parameter sourced or flagged as a
hypothesis): for 1,000 subscribers, ~58 long lies (≥ 1 h on the floor) avoided per year, worth
~138 k EUR. The system breaks even only below **~0.02 false alerts per person per day**: the
daily-life false-alert rate, not measurable on public clips, decides the business case.

**Limits:** 70 training videos from one camera set-up; clear drop on other homes (camera placed
high on Le2i); public daily-life clips last minutes, so no real-life false-alert rate; main error
= bending down quickly, which looks like a fall over 1 s of skeleton.

## Reproduce

Python 3.12, Mac / Linux / Windows, CPU is enough.

```bash
git clone https://github.com/Cdr-23/Fall_detection && cd Fall_detection
python3.12 -m venv .venv && source .venv/bin/activate      # Windows: py -3.12 -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
make all            # Mac/Linux; ~6 min on a laptop: train the 6 models, evaluate, compare, impact
```

`make all` starts from the poses and windows already in the repo (`poses/`, `processed/`): no
download needed. Without `make`, run `python src/run_<algo>.py` then
`python src/evaluate.py results/<algo>_oof.parquet` for `naive rules logreg cart boosting nn`,
then `python src/compare.py`, `src/confirm.py`, `src/confirm_long.py`, `src/impact.py`.

| Step (optional) | Command |
|---|---|
| Raw URFD videos (115 MB) and poses again | `python src/download.py`, `python src/extract_pose.py` |
| GMDCSA-24 external set (1.1 GB) | `python src/download.py --gmdcsa`, then `make gmdcsa` |
| Le2i external set (Kaggle account; `Home_01`, `Home_02` in `data/raw/le2i/`) | `make le2i` |
| Any video → class over time, every model | `python src/demo.py video.mp4 [--labels "Falling[3 to 4]; Lying[4 to 9]"]` |
| Unit tests of the metrics | `python tests/test_metrics.py` |

## Repository

| Path | Content |
|---|---|
| `src/extract_pose.py`, `preprocess.py`, `features.py` | video → poses → windows → features |
| `src/run_<algo>.py` | one model each; writes `results/<algo>_oof.parquet` and `models/<algo>.*` |
| `src/metrics.py`, `evaluate.py`, `compare.py` | shared window and event metrics, comparison |
| `src/confirm.py`, `confirm_long.py` | alert confirmation by lying |
| `src/external.py`, `le2i_convert.py`, `demo.py` | external test sets, team videos, demo |
| `src/impact.py`, `config/impact.yaml` | impact model and sensitivity analysis |
| `results/` | out-of-fold predictions, metrics, figures, notes per part (`p2`-`p5_notes.md`) |

## Data and licences

- **URFD** (training): Kwolek & Kepski, 2014, CC BY-NC-SA 4.0, https://fenix.ur.edu.pl/mkepski/ds/uf.html
- **GMDCSA-24** (external test): Alam et al., *Data in Brief* 2024, CC BY 4.0, https://doi.org/10.5281/zenodo.13354453
- **Le2i FDD** (external test, Home scenes): Charfi et al., 2013, via https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia
- Raw videos are not in the repo (`data/` is git-ignored); extracted poses are.
