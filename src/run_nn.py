"""P4 - Neural network on the raw normalised skeletons (no hand-made features).

Input of a window: processed/X_seq.npy (30 frames x 17 keypoints x (x, y)), already centred on
the mid-hips of each frame and scaled by the torso length of the sequence (preprocess.py), plus
one presence channel per frame (1 when the frame has a pose, 0 when it was left empty).
The presence channel is computed from X_seq itself (empty frames are exactly 0), so training and
predict_proba(feats, X_seq) see the same input.

Two simple variants (CLAUDE.md 7.4, course L8-L9):
  mlp : flatten 30 x 35 -> 128 -> 64 -> 3, ReLU, dropout 0.3
  cnn : 1D convolution over time, 35 channels -> 32 -> 64 (kernel 5), global average pooling -> 3
For each outer fold, both are trained on an inner split BY SEQUENCE of the training folds and
early-stopped on the inner validation sequences; the variant with the best inner PR-AUC of class
`fall` predicts the outer test fold. The test fold is never used for any choice.
Loss: cross-entropy weighted by inverse class frequency. Adam, lr 1e-3, weight decay 1e-4, at
most 50 epochs, early stopping on the inner validation PR-AUC of `fall` (patience 10). Augmentation on the training windows only: left/right mirror (x -> -x, swap the
left/right keypoints). CPU, seeds fixed: the run is reproducible.

Outputs: results/nn_oof.parquet, results/nn_variants.csv (inner and outer scores of both variants
per fold), models/nn.pt (final model on the 70 sequences; variant chosen by mean inner PR-AUC).
Usage: python src/run_nn.py      (~2-4 min on a laptop CPU)
"""
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from common import MODELS, RESULTS, SEED, load_data, oof_frame, save_oof

MODEL_PATH = MODELS / "nn.pt"
MAX_EPOCHS, PATIENCE, LR, BATCH, WEIGHT_DECAY = 50, 10, 1e-3, 64, 1e-4
# COCO order: 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists, 11-12 hips,
# 13-14 knees, 15-16 ankles (odd = left, even = right from 1 to 16)
MIRROR = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
_model = None


def to_input(X_seq):
    """(N, 30, 17, 2) -> (N, 30, 35): 34 coordinates + 1 presence channel per frame."""
    X = np.asarray(X_seq, dtype=np.float32)
    present = (X != 0).any(axis=(2, 3)).astype(np.float32)[..., None]
    return np.concatenate([X.reshape(len(X), X.shape[1], -1), present], axis=2)


def mirror(X_seq):
    X = np.asarray(X_seq, dtype=np.float32)[:, :, MIRROR, :].copy()
    X[..., 0] *= -1
    return X


class MLP(nn.Module):
    def __init__(self, n_in=30 * 35):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, c_in=35):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(c_in, 32, 5, padding=2), nn.ReLU(),
                                  nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
                                  nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3))
        self.out = nn.Linear(64, 3)

    def forward(self, x):                      # x: (N, 30, 35) -> channels first
        return self.out(self.conv(x.transpose(1, 2)))


ARCHS = {"mlp": MLP, "cnn": CNN}


def seed_all(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(y):
    counts = np.bincount(y, minlength=3).astype(np.float32)
    return torch.tensor(len(y) / (3 * np.maximum(counts, 1)))


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(X)), dim=1).numpy()


def train(arch, X_tr, y_tr, X_val, y_val):
    """Train with mirror augmentation, early stopping on the validation PR-AUC of class `fall`
    (the selection metric; the weighted loss proved too noisy with ~30 validation fall windows).
    Returns the best model and the number of epochs it was trained for."""
    seed_all()
    X_aug = np.concatenate([X_tr, mirror(X_tr)])
    y_aug = np.concatenate([y_tr, y_tr])
    xt, yt = torch.tensor(to_input(X_aug)), torch.tensor(y_aug)
    xv = to_input(X_val)
    model = ARCHS[arch]()
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_tr))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(SEED)
    best, best_state, best_epoch, wait = -np.inf, None, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for idx in torch.randperm(len(xt), generator=gen).split(BATCH):
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()
        val = pr_auc_fall(y_val, predict(model, xv))
        if val > best + 1e-4:
            best, best_epoch, wait = val, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_epoch


def inner_split(meta_tr):
    """One split BY SEQUENCE of the training windows (~20 % of the sequences for validation),
    stratified on fall / adl so that the validation set contains falls."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, val = next(sgkf.split(meta_tr, meta_tr["kind"], groups=meta_tr["seq"]))
    return tr, val


def pr_auc_fall(y, proba):
    return average_precision_score(y == 1, proba[:, 1])


def main():
    torch.set_num_threads(4)
    meta, _, X_seq, _ = load_data()
    y = meta["y"].to_numpy()
    proba = np.zeros((len(meta), 3))
    rows, t0 = [], time.time()
    for k in sorted(meta["fold"].unique()):
        tr_idx = np.flatnonzero((meta["fold"] != k).to_numpy())
        te_idx = np.flatnonzero((meta["fold"] == k).to_numpy())
        i_tr, i_val = inner_split(meta.iloc[tr_idx])
        a, v = tr_idx[i_tr], tr_idx[i_val]
        fitted = {}
        for arch in ARCHS:
            model, epochs = train(arch, X_seq[a], y[a], X_seq[v], y[v])
            inner = pr_auc_fall(y[v], predict(model, to_input(X_seq[v])))
            outer = pr_auc_fall(y[te_idx], predict(model, to_input(X_seq[te_idx])))
            fitted[arch] = (inner, model)
            rows.append({"fold": k, "arch": arch, "epochs": epochs, "inner_pr_auc_fall": inner,
                         "outer_pr_auc_fall": outer})
        chosen = max(fitted, key=lambda name: fitted[name][0])   # chosen on the inner split only
        proba[te_idx] = predict(fitted[chosen][1], to_input(X_seq[te_idx]))
        for r in rows[-len(ARCHS):]:
            r["chosen"] = r["arch"] == chosen
        print(f"fold {k}: " + ", ".join(f"{r['arch']} inner {r['inner_pr_auc_fall']:.2f} "
              f"(test {r['outer_pr_auc_fall']:.2f}, {r['epochs']} ep)" for r in rows[-len(ARCHS):])
              + f" -> {chosen}  [{time.time() - t0:.0f}s]", flush=True)

    print("->", save_oof("nn", oof_frame(meta, proba)))
    variants = pd.DataFrame(rows)
    variants.to_csv(RESULTS / "nn_variants.csv", index=False)
    print("->", RESULTS / "nn_variants.csv")

    # Final model on the 70 sequences: variant with the best mean inner PR-AUC, early-stopped on
    # an inner split by sequence (the epoch count needs a validation set).
    arch = variants.groupby("arch")["inner_pr_auc_fall"].mean().idxmax()
    i_tr, i_val = inner_split(meta)
    model, epochs = train(arch, X_seq[i_tr], y[i_tr], X_seq[i_val], y[i_val])
    MODELS.mkdir(exist_ok=True)
    torch.save({"arch": arch, "state_dict": model.state_dict(), "epochs": epochs}, MODEL_PATH)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"-> {MODEL_PATH} (final arch: {arch}, {epochs} epochs, {n_params} parameters)")


def predict_proba(feats: pd.DataFrame, X_seq: np.ndarray) -> np.ndarray:
    """(N, 3) probabilities of not fall / fall / lying from the final model. Uses X_seq only."""
    global _model
    if _model is None:
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        _model = ARCHS[ckpt["arch"]]()
        _model.load_state_dict(ckpt["state_dict"])
        _model.eval()
    return predict(_model, to_input(X_seq))


if __name__ == "__main__":
    main()
