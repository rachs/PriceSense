"""
app.py
------
PriceSense AI – Streamlit application for predicting promotional unit lift.

Run with:
    streamlit run app.py
"""

import os
import random

import streamlit as st

from promo_lift_model import load_models
from predict_promo_lift import predict_promo_lift, VALID_PROMO_TYPES

MODEL_PATH = "promo_models.csv"
MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


@st.cache_resource
def load_model_store():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model_store, promo_lift_df = load_models(MODEL_PATH)
    return model_store, promo_lift_df


@st.cache_data
def get_sample_upcs(_model_store, n: int = 5) -> list:
    """Return n reproducibly-random UPCs from the model store."""
    all_upcs = sorted(_model_store.keys())
    rng = random.Random(42)
    return rng.sample(all_upcs, min(n, len(all_upcs)))


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PriceSense AI",
    page_icon="🏷️",
    layout="centered",
)

st.title("🏷️ PriceSense AI")
st.caption("Predict the promotional unit lift at a given price and calendar month.")

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

model_store, promo_lift_df = load_model_store()

if model_store is None:
    st.error(
        f"Model file **{MODEL_PATH}** not found. "
        "Run `python promo_lift_model.py` first to train and save the models."
    )
    st.stop()

sample_upcs = get_sample_upcs(model_store)

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------

with st.form("promo_lift_form"):
    col1, col2 = st.columns(2)

    with col1:
        selected_upc = st.selectbox(
            "UPC",
            options=sample_upcs,
            format_func=lambda x: str(int(x)) if float(x).is_integer() else str(x),
            help="Select a product UPC (5 random products from the trained model store).",
        )

        promo_type = st.selectbox(
            "Promo Type",
            options=sorted(VALID_PROMO_TYPES),
            help="Type of promotion to evaluate.",
        )

    with col2:
        new_price = st.number_input(
            "New Price ($)",
            min_value=0.01,
            value=2.99,
            step=0.10,
            format="%.2f",
            help="Proposed selling price in dollars.",
        )

        month_label = st.selectbox(
            "Month",
            options=list(MONTH_NAMES.keys()),
            format_func=lambda m: MONTH_NAMES[m],
            index=5,  # default to June
            help="Calendar month for the promotion.",
        )

    submitted = st.form_submit_button("🔮 Predict Lift", use_container_width=True)

# Show base price for the selected UPC
# Load dunnhumby_trans.csv to get the median base_price price for the selected UPC in that month and 
# display it as info text below the form. This gives the user context on how the new price compares to 
# the typical price for that product.
if submitted:
    import pandas as pd
    from pathlib import Path
    data_path = Path(__file__).resolve().parent / "data" / "dunnhumby_trans.csv"
    if data_path.exists():
        trans_df = pd.read_csv(data_path)
        base_price_upc_month = trans_df.loc[
            (trans_df["UPC"] == selected_upc) &
            (pd.to_datetime(trans_df["WEEK_END_DATE"]).dt.month == month_label),
            "BASE_PRICE",
        ].median()
        if pd.notna(base_price_upc_month):
            st.info(f"💰 The median (base) price for UPC **{str(int(selected_upc)) if float(selected_upc).is_integer() else str(selected_upc)}** is **${base_price_upc_month:.2f}**.")
    else:
        st.warning(f"Data file not found: {data_path}. Add 'dunnhumby_trans.csv' under the app's data/ directory to enable base-price display.")

# ---------------------------------------------------------------------------
# Prediction & results
# ---------------------------------------------------------------------------

if submitted:
    result = predict_promo_lift(
        model=model_store[selected_upc],
        upc=selected_upc,
        new_price=float(new_price),
        promo_type=promo_type,
        month=int(month_label),
    )

    if result is None:
        st.error("Prediction failed. Please check your inputs.")
    else:
        st.divider()
        st.subheader("📊 Prediction Results")

        upc_display = str(int(selected_upc)) if float(selected_upc).is_integer() else str(selected_upc)
        st.markdown(
            f"**UPC:** {upc_display} &nbsp;|&nbsp; "
            f"**Promo:** {promo_type} &nbsp;|&nbsp; "
            f"**Price:** ${new_price:.2f} &nbsp;|&nbsp; "
            f"**Month:** {MONTH_NAMES[month_label]}"
        )

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        metric_col1.metric(
            label="Baseline Units",
            value=f"{result['baseline_units']:,.1f}",
            help="Predicted units sold with no promotion active.",
        )
        metric_col2.metric(
            label="Promo Units",
            value=f"{result['promo_units']:,.1f}",
            help="Predicted units sold with the promotion active.",
        )
        metric_col3.metric(
            label="Unit Lift",
            value=f"{result['unit_lift']:+,.1f}",
            delta=f"{result['unit_lift']:+,.1f} units",
            help="Incremental units delivered by the promotion.",
        )
        metric_col4.metric(
            label="Lift %",
            value=f"{result['lift_pct']:+.2f}%",
            delta=f"{result['lift_pct']:+.2f}%",
            help="Percentage increase in units vs. no-promo baseline.",
        )

        if result["lift_pct"] > 0:
            st.success(
                f"✅ The **{promo_type}** promotion at **${new_price:.2f}** in "
                f"**{MONTH_NAMES[month_label]}** is projected to lift units by "
                f"**{result['lift_pct']:+.2f}%** ({result['unit_lift']:+,.1f} units)."
            )
        else:
            st.warning(
                f"⚠️ The **{promo_type}** promotion at **${new_price:.2f}** in "
                f"**{MONTH_NAMES[month_label]}** shows a projected unit change of "
                f"**{result['lift_pct']:+.2f}%** ({result['unit_lift']:+,.1f} units)."
            )
