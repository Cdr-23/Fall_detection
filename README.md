# Fall detection from 2D skeletons

**What it does.** A camera films the home of an older person living alone. Every second, the
system says whether the person is *not falling*, *falling* or *lying on the floor*, and raises an
alert when they fall. Only the body's skeleton is kept, never the image (privacy), and it runs on
an ordinary computer. Target users: telecare operators, who call the person after an alert and
send help if needed.
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

**How to read the numbers.**
- *Falls detected*: out of the real falls in the videos, how many triggered an alert.
- *False alerts*: alerts raised while nobody fell, counted in the daily-life videos (walking,
  sitting, bending, lying on a bed).
- *Delay*: time between the start of the fall and the alert. The confirmation step adds about
  1 second, because the system waits to see the person on the floor; this is negligible next to
  a time on the floor counted in hours.

**1. The 6 models compared** (URFD, 30 falls and 40 daily-life videos; each video is predicted by
a model that never saw it, 5-fold cross-validation):

| Model | Alert only: falls detected | Alert only: false alerts | With confirmation: falls detected | With confirmation: false alerts |
|---|---|---|---|---|
| Naive (always "no fall") | 0 / 30 | 0 | 0 / 30 | 0 |
| Fixed threshold rules | 28 / 30 | 30 | 20 / 30 | 17 |
| Logistic regression | 29 / 30 | 37 | 25 / 30 | 17 |
| Decision tree | 29 / 30 | 35 | 23 / 30 | 16 |
| **Gradient boosting** | **29 / 30** | **15** | **27 / 30** | **7** |
| Neural network | 25 / 30 | 24 | 24 / 30 | 11 |

Mean delay: 0.4-0.6 s with the alert only, 1.3-1.5 s with confirmation.
Confirmation roughly halves the false alerts of every model, at the cost of 1 to 8 missed falls
(only 2 for the boosting).
**Retained: gradient boosting with confirmation**, best on both counts.

**2. The retained model in other homes** (datasets it never saw, gradient boosting with
confirmation):

| Dataset | Falls detected (range) | False alerts | Mean delay |
|---|---|---|---|
| URFD (training homes, cross-validation, for reference) | 27 / 30 (74-97 %) | 7 (40 videos) | 1.4 s |
| GMDCSA-24 (3 other homes) | 52 / 78 (56-76 %) | 9 (81 videos) | 1.4 s |
| Le2i (2 other homes, camera placed high) | 22 / 37 (43-74 %) | 2 (23 videos) | 0.9 s |

In brackets: the 95 % uncertainty range of the detection rate. With so few falls, a difference
of 1 or 2 falls between models is not meaningful.

Performance drops when the room, the camera and the people change: this is the main limit.

**Impact** (`config/impact.yaml` → [`results/impact.md`](results/impact.md); every number has
a source or is marked as a hypothesis). For 1,000 people monitored for a year, about **58 long
stays on the floor (≥ 1 h) avoided** (51 to 78 depending on the scenario). In euros, the result
depends on one unknown, the real-life false-alert rate (each false alert costs ~10 EUR: a check
call, sometimes a dispatch):

| Scenario | False alerts per person | Net value per person per year |
|---|---|---|
| Optimistic | one every 100 days | +291 EUR |
| Central | one every 20 days | −108 EUR |
| Pessimistic | one every 5 days | −1,802 EUR |

**It pays off if false alerts stay below about one per person every 48 days.** The euros only
count part of the hospital cost (not deaths, care-home moves or quality of life), so they are a
lower bound. Next step: a pilot with a telecare operator to measure the real false-alert rate.

**Limits.** 70 training videos, all filmed by the same camera; performance drops in other homes;
the daily-life videos only last a few minutes in total; the main error is bending down quickly,
which looks like a fall over one second of skeleton; the skeleton is scaled using the whole
video, so a live system needs a small change and may do slightly worse. The alert threshold
(0.5) is consistent with the cost ratio of the impact model (a missed fall ≈ 70 false alerts).

More detail: [`results/p5_notes.md`](results/p5_notes.md) (analysis and conclusion),
[`results/comparison.md`](results/comparison.md) (all tables), `results/*.png` (figures).

## Work in progress: telling a fall from lying down on purpose

**Where the false alerts come from.** The model alone mistakes bending down quickly for a fall,
because over one second of skeleton both are the same movement. The confirmation step removes
these: the person gets up again. **The 7 false alerts left on URFD all come from people lying
down on purpose** (on a bed, a sofa or the floor): a quick movement followed by lying down looks
exactly like a fall followed by time on the floor.

**What we are testing.** Two ideas, each giving the model one piece of information it lacks
today:
1. **Height above the floor.** After a fall the body is at floor level; on a bed it is higher.
   The floor level is estimated automatically from the person's ankles while they were standing,
   with no set-up in the home.
2. **A longer look back (3 s instead of 1 s).** Lying down on purpose is controlled: people sit
   first and go down slowly. A fall is sudden and goes all the way down at once.

**How we test it, without fooling ourselves.** With only 7 false alerts left, it would be easy to
tune a fix until they disappear and learn those 7 videos by heart. So the new measures are
defined in advance and never adjusted to the results. The model is retrained exactly as before
(same windows, same 5 folds, same settings), and judged on the two datasets from other homes
(GMDCSA-24, Le2i), on which nothing is tuned. We test each idea alone and both together, so we
know which one helps. The work runs on a separate branch (`p7-context-floor`) and does not touch
the retained model: we will only switch if it clearly helps in the other homes, and we will
report the result either way.

**Not covered.** Someone lying down on purpose on the floor still looks like a fall, and a person
who falls onto their bed looks like someone going to bed. In a deployed system, the operator's
check call before sending anyone remains the safety net; marking the bed and sofa areas when the
camera is installed is a simple product option for the pilot.

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
