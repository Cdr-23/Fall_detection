"""Hyper-parameter tuning shared by run_cart.py and run_boosting.py (P3).

Tuning never sees the outer test fold: for each outer fold k, the grid is scored by an inner
split *by sequence* of the training sequences only (StratifiedGroupKFold, group = seq,
stratified on kind so that every inner fold holds falls). The final model (all 70 sequences)
is tuned the same way on all sequences.

Selection score: PR-AUC (average precision) of the class `fall` (1), the class that triggers
the alert; with ~7 % fall windows it is more informative than accuracy or ROC-AUC.
"""
import itertools

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from common import SEED

N_INNER = 4


def grid(param_grid):
    keys = list(param_grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*param_grid.values())]


def fit(model, X, y, weighted):
    """Fit a Pipeline whose last step is the classifier; `weighted` = balanced sample weights."""
    if weighted:
        step = model.steps[-1][0]
        return model.fit(X, y, **{f"{step}__sample_weight": compute_sample_weight("balanced", y)})
    return model.fit(X, y)


def _score(base, params, X, y, splits, weighted):
    oof = np.zeros(len(y))
    for tr, va in splits:
        m = fit(clone(base).set_params(**params), X.iloc[tr], y[tr], weighted)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return average_precision_score(y == 1, oof)


def tune(base, param_grid, X, y, groups, kinds, weighted=False, n_jobs=-1):
    """Best params by inner CV by sequence. Returns (best_params, table of all scores)."""
    cv = StratifiedGroupKFold(n_splits=N_INNER, shuffle=True, random_state=SEED)
    splits = list(cv.split(X, kinds, groups))
    cands = grid(param_grid)
    scores = Parallel(n_jobs=n_jobs)(
        delayed(_score)(base, p, X, y, splits, weighted) for p in cands)
    table = pd.DataFrame(cands).assign(inner_pr_auc_fall=scores)
    return cands[int(np.argmax(scores))], table


def cross_fit(base, param_grid, meta, X, weighted=False):
    """Outer 5-fold loop of the contract. Returns (oof proba (N,3), per-fold chosen params)."""
    y = meta["y"].to_numpy()
    proba = np.zeros((len(meta), 3))
    chosen = []
    for k in sorted(meta["fold"].unique()):
        tr, te = (meta["fold"] != k).to_numpy(), (meta["fold"] == k).to_numpy()
        best, table = tune(base, param_grid, X[tr], y[tr], meta["seq"][tr], meta["kind"][tr],
                           weighted)
        model = fit(clone(base).set_params(**best), X[tr], y[tr], weighted)
        proba[te] = model.predict_proba(X[te])
        chosen.append({"fold": k, **best, "inner_pr_auc_fall": table["inner_pr_auc_fall"].max()})
        print(f"fold {k}: {best}  inner PR-AUC(fall) = {table['inner_pr_auc_fall'].max():.3f}")
    return proba, pd.DataFrame(chosen)


def fit_final(base, param_grid, meta, X, weighted=False):
    """Final model on all 70 sequences, tuned by inner CV by sequence on all of them."""
    y = meta["y"].to_numpy()
    best, _ = tune(base, param_grid, X, y, meta["seq"], meta["kind"], weighted)
    print(f"final: {best}")
    return fit(clone(base).set_params(**best), X, y, weighted), best
