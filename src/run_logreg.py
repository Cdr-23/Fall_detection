"""Logistic regression (P2, CLAUDE.md section 7.2), ported from Alain's src/p2.py to the shared data.

Pipeline(SimpleImputer(median) -> StandardScaler -> LogisticRegression(class_weight='balanced')),
multinomial, on the 26 per-window features. C in {0.01, 0.1, 1, 10} is chosen by an inner split by
sequence inside the training folds (tree_tuning.py, score = PR-AUC of `fall`), never on the test
fold. Imputation and scaling are fitted on the training windows only (inside the pipeline).

Outputs: results/logreg_oof.parquet, results/logreg_params.csv (chosen C per fold + final),
results/logreg_coefficients.csv (standardised coefficients of the final model, per class),
models/logreg.joblib.
Usage: python src/run_logreg.py
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import FEATURES, MODELS, RESULTS, SEED, load_data, oof_frame, save_oof
from tree_tuning import cross_fit, fit_final

MODEL_PATH = MODELS / "logreg.joblib"
PARAM_GRID = {"logreg__C": [0.01, 0.1, 1, 10]}
_model = None


def make_model():
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("logreg", LogisticRegression(class_weight="balanced", max_iter=5000,
                                                   random_state=SEED))])


def predict_proba(feats: pd.DataFrame, X_seq: np.ndarray) -> np.ndarray:
    """(N, 3) probabilities of not fall / fall / lying, from the final model."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model.predict_proba(feats[FEATURES])


def main():
    meta, feats, _, _ = load_data()
    X = feats[FEATURES]
    proba, chosen = cross_fit(make_model(), PARAM_GRID, meta, X)
    print("->", save_oof("logreg", oof_frame(meta, proba)))

    model, best = fit_final(make_model(), PARAM_GRID, meta, X)
    chosen = pd.concat([chosen, pd.DataFrame([{"fold": "final", **best}])])
    chosen.to_csv(RESULTS / "logreg_params.csv", index=False)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    coef = pd.DataFrame(model.named_steps["logreg"].coef_.T, index=FEATURES,
                        columns=["not_fall", "fall", "lying"])
    coef = coef.sort_values("fall", key=np.abs, ascending=False)
    coef.round(3).to_csv(RESULTS / "logreg_coefficients.csv", index_label="feature")
    print("top coefficients for `fall` (standardised):")
    print(coef["fall"].head(8).round(2).to_string())
    print(f"-> {RESULTS / 'logreg_params.csv'}\n-> {RESULTS / 'logreg_coefficients.csv'}\n-> {MODEL_PATH}")


if __name__ == "__main__":
    main()
