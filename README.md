# Fall detection from 2D skeletons

**What it does.** A camera films the home of an older person living alone. Every second, the
system says whether the person is *not falling*, *falling* or *lying on the floor*, and raises an
alert when they fall. Only the body's skeleton is kept, never the image (privacy), and it runs on
an ordinary computer. Target users: telecare operators and the Paris fire brigade (BSPP).
MIT Sloan 15.071 *The AI Edge*, Project 1, November 2026.

## How it works

1. A ready-made pose model (YOLOv8n-pose) turns each video frame into a skeleton of 17 points.
2. The skeleton is cut into 1-second windows; for each window we compute simple measures:
   torso angle, how fast the hips go down, head height.
3. A model reads these measures and outputs the probability of a fall. We compared 6 models,
   from the simplest (always "no fall", fixed threshold rules) to a neural network.
4. **Two-step alert:** a likely fall is first detected, then **confirmed only if the person is
   then seen lying on the floor**. This removes many false alerts (someone bending down or
   sitting quickly gets up again).

## Results

**Retained model: gradient boosting with the two-step alert.** It was chosen on the training
data (URFD, by cross-validation), then tested on two other datasets filmed in other homes, which
it never saw.

| Dataset | Real falls | Falls detected | False alerts in daily-life videos |
|---|---|---|---|
| URFD (training set, cross-validation) | 30 | **27** | **7** (in 40 videos) |
| GMDCSA-24 (3 other homes) | 78 | **52** | **9** (in 81 videos) |
| Le2i (2 other homes, camera placed high) | 37 | **22** | **2** (in 23 videos) |

A false alert = an alert raised while nobody fell. Without the confirmation step, the same model
detects slightly more falls on URFD and GMDCSA-24 (29 and 53) but raises 2 to 4 times more false
alerts (15, 33 and 7). The confirmation delays the alert by about 1 second (the alert comes 1 to 1.4 s after
the fall starts).

**Comparison of the 6 models** (URFD, alert without confirmation):

| Model | Falls detected (of 30) | False alerts (40 daily-life videos) |
|---|---|---|
| Naive (always "no fall") | 0 | 0 |
| Fixed threshold rules | 28 | 30 |
| Logistic regression | 29 | 37 |
| Decision tree | 29 | 35 |
| **Gradient boosting** | **29** | **15** |
| Neural network | 25 | 24 |

**Impact** (`config/impact.yaml` → `results/impact.md`; every number has a source or is marked
as a hypothesis). For 1,000 people monitored for a year, about 58 long stays on the floor (≥ 1 h)
would be avoided, worth about 138,000 EUR. But it only pays off if false alerts stay rare in daily
life: **below about 1 false alert per person every 7 weeks**. That rate cannot be measured on
public videos, so a pilot with a telecare operator is the next step.

**Limits.** 70 training videos, all filmed by the same camera; performance drops in other homes;
the daily-life videos only last a few minutes in total; the main error is bending down quickly,
which looks like a fall over one second of skeleton.

More detail: [`results/p5_notes.md`](results/p5_notes.md) (analysis and conclusion),
[`results/comparison.md`](results/comparison.md) (all tables), `results/*.png` (figures).

## Reproduce

Python 3.12, Mac / Linux / Windows; no GPU and no download needed (the skeletons are in the repo).

```bash
git clone https://github.com/Cdr-23/Fall_detection && cd Fall_detection
python3.12 -m venv .venv && source .venv/bin/activate   # Windows: py -3.12 -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
make all                                                 # ~6 min: trains the 6 models and rebuilds every result
```

Without `make`, run the commands of the `models`, `compare` and `impact` targets in the
[`Makefile`](Makefile). To test any
video: `python src/demo.py video.mp4`. The external datasets and the team videos are described
in [`CLAUDE.md`](CLAUDE.md) (sections 0 and 5).

## Data

- **URFD** (training): Kwolek & Kepski, 2014, CC BY-NC-SA 4.0, https://fenix.ur.edu.pl/mkepski/ds/uf.html
- **GMDCSA-24** (test): Alam et al., *Data in Brief* 2024, CC BY 4.0, https://doi.org/10.5281/zenodo.13354453
- **Le2i FDD** (test, home scenes): Charfi et al., 2013, via https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia
- Videos are not stored in the repo; the extracted skeletons are (`poses/`).
