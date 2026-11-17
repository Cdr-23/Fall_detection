"""Decision tree (CART) on the per-window features (P3, CLAUDE.md section 7.3).

Pipeline(SimpleImputer(median) -> DecisionTreeClassifier(class_weight='balanced')).
max_depth in 2..6 and min_samples_leaf in {5, 10, 20} are chosen by an inner split by sequence
inside the training folds (see tree_tuning.py), never on the test fold.

Outputs: results/cart_oof.parquet, results/cart_params.csv (chosen per fold + final),
models/cart.joblib, results/cart_tree.png and results/cart_rules.txt (readable final tree).
Usage: python src/run_cart.py
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree  # noqa: E402

from common import FEATURES, MODELS, RESULTS, SEED, load_data, oof_frame, save_oof  # noqa: E402
from tree_tuning import cross_fit, fit_final  # noqa: E402

MODEL_PATH = MODELS / "cart.joblib"
CLASS_NAMES = ["not fall", "fall", "lying"]
PARAM_GRID = {"tree__max_depth": [2, 3, 4, 5, 6], "tree__min_samples_leaf": [5, 10, 20]}
_model = None


def make_model():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("tree", DecisionTreeClassifier(class_weight="balanced", random_state=SEED))])


def predict_proba(feats: pd.DataFrame, X_seq: np.ndarray) -> np.ndarray:
    """(N, 3) probabilities of not fall / fall / lying, from the final model."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model.predict_proba(feats[FEATURES])


def export_tree(model):
    tree = model.named_steps["tree"]
    (RESULTS / "cart_rules.txt").write_text(
        export_text(tree, feature_names=FEATURES, class_names=CLASS_NAMES, decimals=2))
    depth = tree.get_depth()
    fig, ax = plt.subplots(figsize=(min(4 * 2 ** depth, 60), 3 * depth + 2))
    plot_tree(tree, feature_names=FEATURES, class_names=CLASS_NAMES, filled=True, rounded=True,
              impurity=False, proportion=True, precision=2, fontsize=8, ax=ax)
    fig.savefig(RESULTS / "cart_tree.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    meta, feats, _, _ = load_data()
    X = feats[FEATURES]
    proba, chosen = cross_fit(make_model(), PARAM_GRID, meta, X)
    print("->", save_oof("cart", oof_frame(meta, proba)))

    model, best = fit_final(make_model(), PARAM_GRID, meta, X)
    chosen = pd.concat([chosen, pd.DataFrame([{"fold": "final", **best}])], ignore_index=True)
    chosen.to_csv(RESULTS / "cart_params.csv", index=False)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    export_tree(model)
    print("->", MODEL_PATH, RESULTS / "cart_rules.txt", RESULTS / "cart_tree.png")


if __name__ == "__main__":
    main()
