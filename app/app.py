"""Streamlit app: estimate the base rent (Kaltmiete) of a German apartment.

Run from the project root (after `python src/train.py`):
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from cleaning import prepare_features  # noqa: E402

MODEL_PATH = ROOT / "models" / "rent_model.joblib"

# Readable labels for the app (internal value -> what the user sees)
EXTRAS = {
    "balcony": "Balcony",
    "hasKitchen": "Fitted kitchen",
    "lift": "Lift",
    "cellar": "Cellar",
    "garden": "Garden",
    "newlyConst": "Newly constructed",
}
FEATURE_LABELS = {
    "livingSpace": "Living space",
    "noRooms": "Rooms",
    "yearConstructed": "Construction year",
    "regio1": "Bundesland",
    "regio2": "City / district",
    "condition": "Condition",
    "typeOfFlat": "Type of flat",
    "interiorQual": "Interior quality",
    **EXTRAS,
}


def pretty(value):
    """'Baden_Württemberg' -> 'Baden Württemberg', 'first_time_use' -> 'First time use'."""
    text = str(value).replace("_", " ")
    return text[0].upper() + text[1:]


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


def explain(model, X):
    """Effect of each feature on this prediction, in % vs. the average flat.

    LightGBM can return SHAP values directly (pred_contrib=True). They are on the
    log scale, so exp(value) - 1 turns them into a percentage change of the rent.
    """
    contrib = model.regressor_.predict(X, pred_contrib=True)[0]
    effects = pd.Series(contrib[:-1], index=X.columns)   # last value = average prediction
    effects = (np.exp(effects) - 1) * 100
    return np.exp(contrib[-1]), effects


st.set_page_config(page_title="Rent Price Predictor", page_icon="🏠", layout="centered")

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python src/train.py` first.")
    st.stop()

bundle = load_bundle()
model, metrics = bundle["model"], bundle["metrics"]
city_to_state = bundle["city_to_state"]

st.title("🏠 Rent Price Predictor")
st.write(
    "Estimate the monthly **base rent (Kaltmiete)** of an apartment in Germany. "
    f"Trained on {bundle['n_train_rows']:,} ImmoScout24 listings (2018–2020)."
)

# ---------- Inputs ----------
st.subheader("Location")
cities = sorted(city_to_state, key=pretty)
city = st.selectbox("City / district", cities, index=cities.index("Berlin"), format_func=pretty)
st.caption(f"Bundesland: {pretty(city_to_state[city])}")

st.subheader("Apartment")
col1, col2, col3 = st.columns(3)
living_space = col1.number_input("Living space (m²)", min_value=10.0, max_value=300.0, value=70.0, step=5.0)
rooms = col2.number_input("Rooms", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
year_known = col3.checkbox("Year known", value=True)
year = col3.number_input("Construction year", min_value=1800, max_value=2026, value=1990,
                         disabled=not year_known, label_visibility="collapsed")

choices = bundle["choices"]
col1, col2, col3 = st.columns(3)
condition = col1.selectbox("Condition", choices["condition"],
                           index=choices["condition"].index("well_kept"), format_func=pretty)
type_of_flat = col2.selectbox("Type of flat", choices["typeOfFlat"],
                              index=choices["typeOfFlat"].index("apartment"), format_func=pretty)
interior = col3.selectbox("Interior quality", choices["interiorQual"],
                          index=choices["interiorQual"].index("normal"), format_func=pretty)

st.subheader("Extras")
extra_cols = st.columns(3)
extras = {key: extra_cols[i % 3].checkbox(label) for i, (key, label) in enumerate(EXTRAS.items())}

# ---------- Prediction ----------
flat = pd.DataFrame([{
    "livingSpace": living_space,
    "noRooms": rooms,
    "yearConstructed": year if year_known else np.nan,   # LightGBM handles "unknown year"
    **extras,
    "regio1": city_to_state[city],
    "regio2": city,
    "condition": condition,
    "typeOfFlat": type_of_flat,
    "interiorQual": interior,
}])
X = prepare_features(flat)
rent = float(model.predict(X)[0])

# Half of the test predictions were within this % of the true rent
typical_error = metrics["test_median_abs_pct_error"] / 100

st.divider()
st.metric("Estimated base rent", f"€{rent:,.0f} / month")
st.write(
    f"≈ **€{rent / living_space:.2f} per m²**. Typical range: "
    f"€{rent * (1 - typical_error):,.0f} – €{rent * (1 + typical_error):,.0f} "
    f"(half of the test predictions were within ±{typical_error:.0%})."
)

# ---------- Explanation ----------
st.subheader("Why this price?")
average_rent, effects = explain(model, X)
st.write(f"Starting point: the average flat at **€{average_rent:,.0f}**. "
         "Each feature then raises or lowers the estimate:")

effects = effects[effects.abs() >= 0.5].sort_values(key=abs, ascending=False)
table = pd.DataFrame({
    "Feature": [FEATURE_LABELS[f] for f in effects.index],
    "Effect on rent": [f"{v:+.1f}%" for v in effects.values],
})
st.dataframe(table, hide_index=True, width="stretch")
st.caption("Effects multiply: +10% and then +10% is +21%. They show what the model learned "
           "from the data, not necessarily what causes higher rents.")

with st.expander("About the model"):
    st.write(
        f"LightGBM (gradient boosted trees) trained on log(rent). On a held-out test set: "
        f"mean absolute error **€{metrics['test_mae_eur']:.0f}**, "
        f"median error **{metrics['test_median_abs_pct_error']}%**, R² **{metrics['test_r2']}**. "
        "Data: Kaggle 'Apartment rental offers in Germany' (ImmoScout24, 2018–2020) – "
        "today's rents are likely higher."
    )
