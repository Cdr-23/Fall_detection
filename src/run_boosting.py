"""Gradient boosting on the per-window features, with SHAP (P3, CLAUDE.md section 7.3).

Pipeline(SimpleImputer(median) -> GradientBoostingClassifier). GradientBoostingClassifier has no
class_weight: classes are balanced with sample weights (as class_weight='balanced' in the tree),
which shifts p1 upward (moderately, see run_boosting_unweighted.py); tau is swept by P5.
Grid: n_estimators {100, 200, 300} x max_depth {2, 3} x learning_rate {0.05, 0.1}, chosen by an
inner split by sequence inside the training folds (see tree_tuning.py), never on the test fold.

Outputs: results/boosting_oof.parquet, results/boosting_params.csv, models/boosting.joblib,
results/shap_summary.png (class `fall`) and results/shap_importance.csv (mean |SHAP| per
feature and class, final model on all 70 sequences).
Usage: python src/run_boosting.py [--shap-only]
"""
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from common import FEATURES, MODELS, RESULTS, SEED, load_data, oof_frame, save_oof  # noqa: E402
from tree_tuning import cross_fit, fit_final  # noqa: E402

MODEL_PATH = MODELS / "boosting.joblib"
PARAM_GRID = {"gb__n_estimators": [100, 200, 300], "gb__max_depth": [2, 3],
              "gb__learning_rate": [0.05, 0.1]}
_model = None


def make_model():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("gb", GradientBoostingClassifier(random_state=SEED))])


def predict_proba(feats: pd.DataFrame, X_seq: np.ndarray) -> np.ndarray:
    """(N, 3) probabilities of not fall / fall / lying, from the final model."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model.predict_proba(feats[FEATURES])


def explain(model, X):
    """SHAP on the final model (raw score per class), on the median-imputed features.

    shap.TreeExplainer only supports a binary GradientBoostingClassifier. The multiclass raw
    score of class k is init_k + learning_rate * sum of the regression trees estimators_[:, k],
    and SHAP values are additive, so we explain each tree and sum: this is exact.
    """
    gb = model.named_steps["gb"]
    Xi = pd.DataFrame(model.named_steps["impute"].transform(X), columns=FEATURES)
    sv = np.zeros((len(Xi), len(FEATURES), gb.n_classes_))
    for k in range(gb.n_classes_):
        for tree in gb.estimators_[:, k]:
            sv[:, :, k] += shap.TreeExplainer(tree).shap_values(Xi.to_numpy())
    sv *= gb.learning_rate  # (N, F, 3)
    imp = pd.DataFrame(np.abs(sv).mean(axis=0), index=FEATURES,
                       columns=["shap_notfall", "shap_fall", "shap_lying"])
    imp.index.name = "feature"
    imp.sort_values("shap_fall", ascending=False).to_csv(RESULTS / "shap_importance.csv")

    shap.summary_plot(sv[:, :, 1], Xi, max_display=15, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP values for class 'fall' (gradient boosting, all 70 sequences)")
    fig.savefig(RESULTS / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return imp


def main():
    meta, feats, _, _ = load_data()
    X = feats[FEATURES]
    if "--shap-only" in sys.argv:  # re-explain the saved final model without retraining
        explain(joblib.load(MODEL_PATH), X)
        print("->", RESULTS / "shap_summary.png", RESULTS / "shap_importance.csv")
        return
    proba, chosen = cross_fit(make_model(), PARAM_GRID, meta, X, weighted=True)
    print("->", save_oof("boosting", oof_frame(meta, proba)))

    model, best = fit_final(make_model(), PARAM_GRID, meta, X, weighted=True)
    chosen = pd.concat([chosen, pd.DataFrame([{"fold": "final", **best}])], ignore_index=True)
    chosen.to_csv(RESULTS / "boosting_params.csv", index=False)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("->", MODEL_PATH)

    imp = explain(model, X)
    print(imp.sort_values("shap_fall", ascending=False).head(10).round(3))
    print("->", RESULTS / "shap_summary.png", RESULTS / "shap_importance.csv")


if __name__ == "__main__":
    main()
