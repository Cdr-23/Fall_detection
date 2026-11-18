# P4 — Neural network on raw skeletons

Input: `processed/X_seq.npy` only (30 frames × 17 keypoints × (x, y), centred on the mid-hips of
each frame, scaled by the torso length of the sequence) + 1 presence channel per frame. No
hand-made features. Same 5 outer folds by sequence as everyone (contract 7.0).

Two variants (course L8-L9): **MLP** (flatten → 128 → 64 → 3, dropout 0.3) and **1D-CNN** over
time (35 → 32 → 64 channels, kernel 5, global average pooling). In each outer fold, both are
trained on an inner split **by sequence** of the training folds, early-stopped on the inner
validation sequences, and the one with the best inner PR-AUC of `fall` predicts the test fold.
Class-weighted cross-entropy, Adam (lr 1e-3, weight decay 1e-4), ≤ 50 epochs, mirror
augmentation (left/right) on training windows only, CPU, seeds fixed (reproducible).

Run: `python src/run_nn.py` (~1-2 min on a laptop CPU), then
`python src/evaluate.py results/nn_oof.parquet`. Per-fold details: `results/nn_variants.csv`.

## Results (URFD, out-of-fold, tau = 0.5; mean ± std over the 5 folds)

| | naive | CART | **NN (P4)** | Boosting (balanced) |
|---|---|---|---|---|
| ROC-AUC `fall` | 0.50 | 0.83 ± 0.06 | **0.88 ± 0.08** | 0.91 ± 0.08 |
| PR-AUC `fall` | 0.075 | 0.33 ± 0.10 | **0.60 ± 0.13** | 0.67 ± 0.10 |
| F1 `fall` | 0 | 0.47 ± 0.13 | **0.48 ± 0.13** | 0.65 ± 0.08 |
| F1 `lying` | 0 | 0.86 ± 0.02 | **0.88 ± 0.05** | 0.90 ± 0.01 |
| Falls detected (events) | 0 / 30 | 29 / 30 | **25 / 30** | 29 / 30 |
| Mean delay (s) | — | 0.37 | **0.38** | 0.37 |
| False alerts in ADL (raw count) | 0 | 35 | **24** | 15 |
| Latency per window (ms) | — | 0.8 | **0.2** | 1.2 |
| Model size (KB) | — | 7 | **561** | 1 128 |

CART and boosting columns copied from `results/p3_notes.md` (branch `p3-trees`); P5 should
recompute everything with `src/metrics.py`.

**Reading:** the network clearly beats the naive model and CART on ranking (PR-AUC 0.60 vs
0.33), but stays below the boosting (0.67) and misses more falls (25 / 30 vs 29 / 30) at
`tau = 0.5`. The gap with the boosting is within one fold std, so with 70 sequences we cannot
claim it is significant. Recognising `lying` is easy for every model (F1 ≈ 0.88).

## Why it does not beat the boosting (likely cause)

`X_seq` is centred on the mid-hips **of each frame**, so the network sees the body shape over
time but **not the global movement of the body** (how fast the hips go down). The boosting gets
that from the features `hip_vy_*` / `hip_ay_*`, which are among its top SHAP features
(`hip_ay_std` 4th, `hip_vy_min` 5th). A fall and a fast sit-down can look alike in shape only.
Possible next step (not done, would change the P4 input): add the hip trajectory of the window
(relative to its first frame) as 2 extra channels. It needs P1 to export it, because
`predict_proba(feats, X_seq)` only receives these two inputs.

## Variants and stability

- Variant chosen per fold (inner split only): MLP in folds 1, 2, 4; CNN in folds 0, 3. Final
  model (`models/nn.pt`): **MLP**, 143 k parameters, best mean inner PR-AUC (0.71 vs 0.64).
- Inner scores are optimistic (the same inner split is used for early stopping and for choosing
  the variant); only the out-of-fold numbers above count.
- **One revision was made.** v1 early-stopped on the weighted validation loss: with ~30
  validation fall windows it was noisy and the CNN often stopped after 3-6 epochs. v2 (kept)
  early-stops on the validation PR-AUC of `fall` and adds weight decay. Out-of-fold, the two
  versions are equivalent — v1: PR-AUC 0.58 ± 0.17, 26 / 30 falls, 23 false alerts; v2:
  0.60 ± 0.13, 25 / 30, 24. v2 is kept for its steadier training, not for a better test score.
- Large spread across folds (PR-AUC from 0.58 to 0.79 per fold): 70 sequences is small, a
  neural network is the most sensitive model to it.
