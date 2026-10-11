# ============================================================
# BANKSENSE AI
# Bank Marketing Subscription Prediction
# ============================================================

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# 1. Application configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="BankSense AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "banksense_random_forest.joblib"
METADATA_PATH = ARTIFACT_DIR / "banksense_metadata.json"


# ------------------------------------------------------------
# 2. Load trained model and metadata
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return model, metadata


st.title("🏦 BankSense AI")
st.subheader("Bank Marketing Subscription Prediction")

st.write(
    """
    Analyze customer and campaign information to estimate the likelihood
    that a customer will subscribe to a bank term deposit.
    """
)

st.caption(
    "Machine-learning demonstration using a trained Random Forest classifier."
)


if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error(
        "Model artifacts were not found. Place the trained model and "
        "metadata inside the artifacts/ directory."
    )
    st.stop()


try:
    model, metadata = load_model()
except Exception as exc:
    st.error(
        "The model could not be loaded. Check the model file and "
        "the installed scikit-learn version."
    )
    st.exception(exc)
    st.stop()


THRESHOLD = float(metadata["decision_threshold"])
EXPECTED_FEATURES = metadata["input_features"]


# ------------------------------------------------------------
# 3. Sidebar
# ------------------------------------------------------------

with st.sidebar:
    st.title("BankSense AI")

    st.markdown("---")

    st.markdown("### Model Information")

    st.write("**Algorithm:** Random Forest")
    st.write(f"**Decision threshold:** {THRESHOLD:.2f}")
    st.write("**Task:** Binary classification")

    st.markdown("---")

    st.markdown("### Important Note")

    st.info(
        "This application provides model-based estimates. "
        "It should not be treated as a guarantee of customer behavior."
    )


# ------------------------------------------------------------
# 4. Customer input form
# ------------------------------------------------------------

st.markdown("## Enter Customer Details")

with st.form("customer_prediction_form"):

    st.markdown("### Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1,
        )

    with col2:
        job = st.selectbox(
            "Job",
            [
                "admin.",
                "blue-collar",
                "entrepreneur",
                "housemaid",
                "management",
                "retired",
                "self-employed",
                "services",
                "student",
                "technician",
                "unemployed",
                "unknown",
            ],
        )

    with col3:
        marital = st.selectbox(
            "Marital Status",
            ["married", "single", "divorced"],
        )

    col4, col5 = st.columns(2)

    with col4:
        education = st.selectbox(
            "Education",
            [
                "primary",
                "secondary",
                "tertiary",
                "unknown",
            ],
        )

    with col5:
        balance = st.number_input(
            "Average Yearly Balance",
            min_value=-100000,
            max_value=1000000,
            value=1500,
            step=100,
        )

    st.markdown("---")
    st.markdown("### Financial Information")

    col6, col7, col8 = st.columns(3)

    with col6:
        default = st.selectbox(
            "Credit in Default?",
            ["no", "yes"],
        )

    with col7:
        housing = st.selectbox(
            "Housing Loan?",
            ["no", "yes"],
        )

    with col8:
        loan = st.selectbox(
            "Personal Loan?",
            ["no", "yes"],
        )

    st.markdown("---")
    st.markdown("### Campaign Information")

    col9, col10, col11 = st.columns(3)

    with col9:
        contact = st.selectbox(
            "Contact Method",
            ["cellular", "telephone", "unknown"],
        )

    with col10:
        day = st.number_input(
            "Last Contact Day",
            min_value=1,
            max_value=31,
            value=15,
            step=1,
        )

    with col11:
        month = st.selectbox(
            "Last Contact Month",
            [
                "jan", "feb", "mar", "apr",
                "may", "jun", "jul", "aug",
                "sep", "oct", "nov", "dec",
            ],
            index=4,
        )

    col12, col13 = st.columns(2)

    with col12:
        campaign = st.number_input(
            "Contacts During Current Campaign",
            min_value=1,
            max_value=100,
            value=2,
            step=1,
        )

    with col13:
        previous = st.number_input(
            "Contacts During Previous Campaigns",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
        )

    st.markdown("---")
    st.markdown("### Previous Campaign History")

    col14, col15 = st.columns(2)

    with col14:
        previously_contacted = st.selectbox(
            "Previously Contacted?",
            ["no", "yes"],
        )

    with col15:
        poutcome = st.selectbox(
            "Previous Campaign Outcome",
            [
                "unknown",
                "failure",
                "other",
                "success",
            ],
        )

    if previously_contacted == "yes":
        pdays = st.number_input(
            "Days Since Previous Contact",
            min_value=0,
            max_value=1000,
            value=30,
            step=1,
        )
    else:
        pdays = -1

    submitted = st.form_submit_button(
        "Predict Subscription",
        type="primary",
        use_container_width=True,
    )


# ------------------------------------------------------------
# 5. Generate prediction
# ------------------------------------------------------------

if submitted:

    # Match the feature names used during training.
    customer_data = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "Previously_Contacted": (
            1 if previously_contacted == "yes" else 0
        ),
    }

    input_df = pd.DataFrame([customer_data])

    # Fail safely if the application and model schemas differ.
    missing_features = set(EXPECTED_FEATURES) - set(input_df.columns)
    extra_features = set(input_df.columns) - set(EXPECTED_FEATURES)

    if missing_features or extra_features:

        st.error("Input features do not match the trained model.")

        if missing_features:
            st.write("Missing features:", sorted(missing_features))

        if extra_features:
            st.write("Unexpected features:", sorted(extra_features))

        st.stop()

    # Match the exact feature order expected by the pipeline.
    input_df = input_df[EXPECTED_FEATURES]

    try:
        probability = float(
            model.predict_proba(input_df)[0, 1]
        )

        prediction = int(
            probability >= THRESHOLD
        )

        st.markdown("---")
        st.markdown("## Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            if prediction == 1:
                st.success("Likely to Subscribe")
                st.write(
                    "The estimated probability meets or exceeds "
                    "the configured decision threshold."
                )
            else:
                st.warning("Unlikely to Subscribe")
                st.write(
                    "The estimated probability is below "
                    "the configured decision threshold."
                )

        with result_col2:

            st.metric(
                "Estimated Subscription Probability",
                f"{probability:.1%}",
            )

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

            st.caption(
                f"Decision threshold: {THRESHOLD:.2f}"
            )

        st.markdown("---")

        st.markdown("### Prediction Details")

        details = pd.DataFrame(
            {
                "Metric": [
                    "Predicted Class",
                    "Estimated Probability",
                    "Decision Threshold",
                    "Model",
                ],
                "Value": [
                    (
                        "Subscribed"
                        if prediction == 1
                        else "Not Subscribed"
                    ),
                    f"{probability:.4f}",
                    f"{THRESHOLD:.2f}",
                    metadata["model_name"],
                ],
            }
        )

        st.dataframe(
            details,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "This is a statistical estimate based on historical data. "
            "It does not guarantee actual customer behavior."
        )

    except Exception as exc:

        st.error(
            "Prediction failed. Verify that the input categories and "
            "feature schema match the training dataset."
        )

        st.exception(exc)


# ------------------------------------------------------------
# 6. Footer
# ------------------------------------------------------------

st.markdown("---")

st.caption(
    "BankSense AI | Machine Learning Project | "
    "Bank Marketing Subscription Prediction"
)