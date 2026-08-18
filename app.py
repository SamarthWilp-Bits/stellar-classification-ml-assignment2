"""
Streamlit app: upload a CSV of held-out SDSS17 observations, pick one of the
6 trained classifiers, and see how it does — accuracy, AUC, precision,
recall, F1, MCC, confusion matrix, and a full classification report.

Models are trained offline by model/train_models.py and loaded here with
joblib; this app never calls .fit() on anything.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from model.metrics_utils import FEATURE_COLUMNS, TARGET_COLUMN, compute_all_metrics

MODEL_DIR = Path(__file__).resolve().parent / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "k-nearest_neighbors.pkl",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
}


@st.cache_resource
def load_artifacts():
    """Load every saved model + the shared preprocessing objects exactly once per app session."""
    missing = [name for name, fname in MODEL_FILES.items() if not (MODEL_DIR / fname).exists()]
    if missing:
        st.error(
            "Missing trained model file(s): "
            + ", ".join(missing)
            + f". Run `python model/train_models.py` first to populate {MODEL_DIR}/."
        )
        st.stop()

    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}
    feature_scaler = joblib.load(MODEL_DIR / "feature_scaler.pkl")
    class_encoder = joblib.load(MODEL_DIR / "class_encoder.pkl")
    return models, feature_scaler, class_encoder


def validate_and_prepare(uploaded_df: pd.DataFrame, class_encoder):
    """
    Check the uploaded CSV actually looks like our held-out test data.
    Returns (object_features, encoded_true_class) on success, or raises
    ValueError with a message meant to be shown directly to the user.
    """
    missing_columns = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in uploaded_df.columns]
    if missing_columns:
        raise ValueError(
            f"CSV is missing required column(s): {missing_columns}. "
            f"Expected all of: {FEATURE_COLUMNS + [TARGET_COLUMN]}."
        )

    object_features = uploaded_df[FEATURE_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        object_features[column] = pd.to_numeric(object_features[column], errors="coerce")
    if object_features.isna().any().any():
        bad_columns = object_features.columns[object_features.isna().any()].tolist()
        raise ValueError(
            f"Column(s) {bad_columns} contain non-numeric or missing values after upload. "
            f"Fix or remove those rows and re-upload."
        )

    unknown_labels = set(uploaded_df[TARGET_COLUMN].unique()) - set(class_encoder.classes_)
    if unknown_labels:
        raise ValueError(
            f"Unknown class label(s) {sorted(unknown_labels)} in the '{TARGET_COLUMN}' column. "
            f"Expected one of: {list(class_encoder.classes_)}."
        )

    encoded_true_class = class_encoder.transform(uploaded_df[TARGET_COLUMN])
    return object_features, encoded_true_class


def main():
    st.set_page_config(page_title="Stellar Classification (SDSS17)", layout="wide")
    st.title("Stellar Classification — SDSS17")
    st.caption("Upload held-out observations, pick a model, inspect its performance.")

    models, feature_scaler, class_encoder = load_artifacts()

    uploaded_file = st.file_uploader(
        "Upload test CSV (must include the original 14 feature columns + 'class' column — "
        "test_data.csv in this repo is a ready-made example)",
        type=["csv"],
    )
    model_name = st.selectbox("Model", list(models.keys()))

    if uploaded_file is None:
        st.info("Upload a CSV to evaluate a model. Try the included test_data.csv.")
        return

    try:
        uploaded_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read that file as a CSV: {exc}")
        return

    try:
        object_features, encoded_true_class = validate_and_prepare(uploaded_df, class_encoder)
    except ValueError as exc:
        st.error(str(exc))
        return

    scaled_features = feature_scaler.transform(object_features)
    model = models[model_name]
    predicted_class = model.predict(scaled_features)
    predicted_probabilities = model.predict_proba(scaled_features)

    class_label_ids = np.arange(len(class_encoder.classes_))
    metrics = compute_all_metrics(encoded_true_class, predicted_class, predicted_probabilities, class_label_ids)

    st.subheader(f"Metrics — {model_name}")
    metric_cols = st.columns(6)
    for col, (metric_name, value) in zip(metric_cols, metrics.items()):
        col.metric(metric_name, "N/A" if (value is None or np.isnan(value)) else f"{value:.4f}")
    if np.isnan(metrics.get("AUC", np.nan)):
        st.caption(
            "AUC shows N/A because the uploaded CSV doesn't contain all 3 classes "
            "(GALAXY/QSO/STAR) — ovr AUC needs every class represented to score."
        )

    left, right = st.columns(2)

    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(encoded_true_class, predicted_class, labels=class_label_ids)
        fig, ax = plt.subplots()
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_encoder.classes_, yticklabels=class_encoder.classes_, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with right:
        st.subheader("Classification report")
        report_dict = classification_report(
            encoded_true_class, predicted_class,
            labels=class_label_ids, target_names=class_encoder.classes_,
            output_dict=True, zero_division=0,
        )
        st.dataframe(pd.DataFrame(report_dict).transpose().round(4))


if __name__ == "__main__":
    main()
