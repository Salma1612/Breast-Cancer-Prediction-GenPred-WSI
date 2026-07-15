"""
app.py
------
A lightweight Streamlit web application that lets a user enter the 30
diagnostic features of a breast tumor (or upload a CSV of samples) and get
a real-time benign/malignant prediction from the trained MLP model.

Run with:
    streamlit run app.py

Requires that `models/trained_model.h5`, `models/scaler.pkl`, and
`models/label_encoder.pkl` already exist (see README.md -> "How to Run").

Author: SK Salma, P. Durgamma, Nithiasri S
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_preprocessing import load_preprocessors, clean_dataset  # noqa: E402
from predict import FEATURE_NAMES, DEMO_SAMPLE, predict  # noqa: E402

MODEL_PATH = "models/trained_model.h5"
PREPROCESSORS_DIR = "models"

st.set_page_config(
    page_title="Breast Cancer Prediction (Deep Learning)",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    """Load the trained Keras model and fitted preprocessors (cached)."""
    from tensorflow import keras
    model = keras.models.load_model(MODEL_PATH)
    scaler, encoder = load_preprocessors(PREPROCESSORS_DIR)
    return model, scaler, encoder


def render_sidebar():
    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses a Multi-Layer Perceptron (MLP) trained on the "
        "**Breast Cancer Wisconsin Diagnostic Dataset (WDBC)** to predict "
        "whether a tumor is **benign** or **malignant** from 30 "
        "digitized fine-needle aspirate (FNA) features.\n\n"
        "Reported test performance: **97.2% accuracy, AUC = 0.98**."
    )
    st.sidebar.warning(
        "⚠️ This tool is a student research project and an educational demo. "
        "It is **not** a certified medical device and must not be used for "
        "real clinical diagnosis."
    )


def manual_input_form():
    st.subheader("Enter Tumor Features Manually")
    st.caption("Defaults are pre-filled with a sample record from the dataset.")

    cols = st.columns(3)
    values = {}
    for i, feature in enumerate(FEATURE_NAMES):
        col = cols[i % 3]
        default = float(DEMO_SAMPLE[feature])
        values[feature] = col.number_input(
            feature.replace("_", " ").title(), value=default, format="%.5f"
        )
    return pd.DataFrame([values])


def csv_upload_form():
    st.subheader("Upload a CSV of Samples")
    st.caption("The CSV should contain the 30 WDBC feature columns "
               "(id / diagnosis columns are ignored if present).")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        df = clean_dataset(df)
        missing = [c for c in FEATURE_NAMES if c not in df.columns]
        if missing:
            st.error(f"Uploaded file is missing required columns: {missing}")
            return None
        return df[FEATURE_NAMES]
    return None


def main():
    st.title("🩺 Breast Cancer Prediction using Deep Learning")
    st.write(
        "A Multi-Layer Perceptron (MLP) classifier for predicting benign vs. "
        "malignant breast tumors, built on the WDBC dataset."
    )
    render_sidebar()

    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Trained model not found at `{MODEL_PATH}`.\n\n"
            "Please run `python src/train.py` first to train and save the model "
            "(see README.md -> 'How to Run')."
        )
        return

    model, scaler, encoder = load_artifacts()

    tab1, tab2 = st.tabs(["Manual Input", "Upload CSV"])
    with tab1:
        input_df = manual_input_form()
        run_manual = st.button("Predict", type="primary", key="manual_predict")
        if run_manual:
            probs, labels = predict(model, scaler, input_df[FEATURE_NAMES].values)
            _render_result(probs[0], labels[0])

    with tab2:
        uploaded_df = csv_upload_form()
        if uploaded_df is not None and st.button("Predict Batch", type="primary"):
            probs, labels = predict(model, scaler, uploaded_df.values)
            result_df = uploaded_df.copy()
            result_df["P(malignant)"] = probs
            result_df["Prediction"] = labels
            st.dataframe(result_df)
            st.download_button(
                "Download Results as CSV",
                result_df.to_csv(index=False).encode("utf-8"),
                "predictions.csv",
                "text/csv",
            )


def _render_result(probability: float, label: str):
    if label == "Malignant":
        st.error(f"### Prediction: {label}")
    else:
        st.success(f"### Prediction: {label}")
    st.metric("Predicted Probability of Malignancy", f"{probability:.2%}")
    st.progress(min(max(float(probability), 0.0), 1.0))


if __name__ == "__main__":
    main()
