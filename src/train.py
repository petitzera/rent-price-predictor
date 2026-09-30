"""Train the final rent model and save it to models/rent_model.joblib.

Run from the project root:
    python src/train.py

Steps:
1. Load and clean the data (same rules as the notebooks, from cleaning.py)
2. Evaluate the model on a held-out 20% test set and print the scores
3. Retrain on ALL data (more data = slightly better model) and save it for the app
"""

from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from cleaning import FEATURES, TARGET, clean_data, load_raw, prepare_features

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "immo_data.csv"
MODEL_PATH = ROOT / "models" / "rent_model.joblib"

# Text options shown in the app's dropdowns
APP_CHOICE_COLUMNS = ["condition", "typeOfFlat", "interiorQual"]


def build_model():
    """LightGBM on log(rent): the best model from 02_modeling.ipynb.

    TransformedTargetRegressor takes the log of the rent before training and
    converts predictions back to euros with exp(), so we never have to do it by hand.
    """
    lgbm = LGBMRegressor(
        n_estimators=1000,    # number of small trees added one after another
        learning_rate=0.05,   # how much each new tree is allowed to correct
        num_leaves=63,        # how detailed each tree can be
        random_state=42,
        verbose=-1,
    )
    return TransformedTargetRegressor(regressor=lgbm, func=np.log, inverse_func=np.exp)


def main():
    df = clean_data(load_raw(DATA_PATH))
    X, y = prepare_features(df), df[TARGET]
    print(f"Clean data: {len(df):,} apartments, {X.shape[1]} features")

    # 1) Honest evaluation on data the model has never seen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pred = build_model().fit(X_train, y_train).predict(X_test)
    metrics = {
        "test_mae_eur": round(mean_absolute_error(y_test, pred), 1),
        "test_median_abs_pct_error": round(float(np.median(np.abs(pred - y_test) / y_test)) * 100, 1),
        "test_r2": round(r2_score(y_test, pred), 3),
    }
    print("Test scores:", metrics)

    # 2) Final model on all data, saved together with what the app needs
    final_model = build_model().fit(X, y)
    bundle = {
        "model": final_model,
        "features": FEATURES,
        "metrics": metrics,
        "n_train_rows": len(X),
        "city_to_state": df.groupby("regio2")["regio1"].first().to_dict(),
        "choices": {col: sorted(X[col].cat.categories) for col in APP_CHOICE_COLUMNS},
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    print(f"Saved model to {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
