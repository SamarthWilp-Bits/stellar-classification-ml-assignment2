"""
Train and evaluate 6 classifiers on the SDSS17 stellar classification dataset
(STAR / GALAXY / QSO), then persist everything the Streamlit app needs to
score new data without ever retraining.

Usage:
    python model/train_models.py
    python model/train_models.py --data-path data/star_classification.csv --output-dir model

Expects the raw Kaggle CSV at data/star_classification.csv (see README for
the download step) with the standard SDSS17 columns.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.metrics_utils import (
    BAD_MAGNITUDE_THRESHOLD,
    FEATURE_COLUMNS,
    ID_COLUMNS_TO_DROP,
    TARGET_COLUMN,
    compute_all_metrics,
    metrics_to_row,
)

RANDOM_STATE = 42
TEST_SET_FRACTION = 0.20         # used for the metrics printed below (needs to be large enough to be reliable)
DEMO_TEST_CSV_ROWS = 500         # small stratified sample of the test split saved as test_data.csv for the repo/app


def load_and_clean(data_path: Path) -> pd.DataFrame:
    stellar_observations = pd.read_csv(data_path)

    missing_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(stellar_observations.columns)
    if missing_columns:
        raise ValueError(f"Raw CSV is missing expected columns: {sorted(missing_columns)}")

    stellar_observations = stellar_observations.drop(columns=ID_COLUMNS_TO_DROP, errors="ignore")

    # Drop rows with the -9999 sentinel / physically-impossible magnitudes
    # (see BAD_MAGNITUDE_THRESHOLD in metrics_utils.py for why).
    magnitude_columns = ["u", "g", "r", "i", "z"]
    is_bad_row = (stellar_observations[magnitude_columns] < BAD_MAGNITUDE_THRESHOLD).any(axis=1)
    n_bad = int(is_bad_row.sum())
    if n_bad:
        print(f"Dropping {n_bad} row(s) with sentinel/invalid magnitude values (< {BAD_MAGNITUDE_THRESHOLD}).")
    stellar_observations = stellar_observations.loc[~is_bad_row].reset_index(drop=True)

    return stellar_observations


def build_models() -> dict:
    """
    All 6 classifiers, using the same preprocessed (scaled) features.

    Naive Bayes choice: GaussianNB, not MultinomialNB. MultinomialNB expects
    non-negative, count-like features (e.g. word counts); our features are
    continuous real-valued measurements (angles, magnitudes, redshift) and
    `delta` (declination) can be negative outright, which MultinomialNB
    cannot accept at all. GaussianNB models each feature as normally
    distributed per class, which is the right fit for continuous
    astronomical measurements.

    GradientBoostingClassifier is the optional 6th model (the brief allows
    SVM or Gradient Boosting as a bonus). Delete it from this dict if your
    assignment wants exactly 5 — everything else in the pipeline adapts
    automatically since it just iterates over this dict.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Gaussian Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/star_classification.csv", type=Path)
    parser.add_argument("--output-dir", default="model", type=Path)
    parser.add_argument("--test-csv-path", default="test_data.csv", type=Path,
                         help="Where to write the small held-out demo CSV used by the Streamlit app.")
    args = parser.parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {args.data_path}. Download star_classification.csv from Kaggle "
            f"(fedesoriano/stellar-classification-dataset-sdss17) and place it there. See README.md."
        )

    stellar_observations = load_and_clean(args.data_path)
    print(f"Loaded {len(stellar_observations)} clean observations, {len(FEATURE_COLUMNS)} features, "
          f"target classes: {sorted(stellar_observations[TARGET_COLUMN].unique())}")

    object_features = stellar_observations[FEATURE_COLUMNS]
    class_encoder = LabelEncoder()
    object_class = class_encoder.fit_transform(stellar_observations[TARGET_COLUMN])

    # Stratified split: GALAXY/STAR/QSO are imbalanced (~45%/16%/14%), so a
    # plain random split risks under-representing QSO/STAR in the test set.
    (
        features_train, features_test,
        class_train, class_test,
        raw_train, raw_test,
    ) = train_test_split(
        object_features, object_class, stellar_observations,
        test_size=TEST_SET_FRACTION, random_state=RANDOM_STATE, stratify=object_class,
    )

    # Fit scaling on the TRAINING split only, then apply the same transform
    # to the test split — fitting on test data would leak test-set
    # statistics into preprocessing and inflate reported performance.
    feature_scaler = StandardScaler()
    features_train_scaled = feature_scaler.fit_transform(features_train)
    features_test_scaled = feature_scaler.transform(features_test)

    # Small stratified sample of the (already held-out) test split, saved
    # with the original unscaled values and the true `class` label — this is
    # what a user uploads to the Streamlit app to see it evaluate a model.
    demo_test_df, _ = train_test_split(
        raw_test[FEATURE_COLUMNS + [TARGET_COLUMN]],
        train_size=min(DEMO_TEST_CSV_ROWS, len(raw_test)),
        random_state=RANDOM_STATE,
        stratify=raw_test[TARGET_COLUMN],
    )
    demo_test_df.to_csv(args.test_csv_path, index=False)
    print(f"Saved {len(demo_test_df)}-row held-out demo CSV to {args.test_csv_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_scaler, args.output_dir / "feature_scaler.pkl")
    joblib.dump(class_encoder, args.output_dir / "class_encoder.pkl")
    joblib.dump(FEATURE_COLUMNS, args.output_dir / "feature_columns.pkl")

    models = build_models()
    comparison_rows = []
    class_label_ids = np.arange(len(class_encoder.classes_))

    for model_name, model in models.items():
        model.fit(features_train_scaled, class_train)

        predicted_class = model.predict(features_test_scaled)
        predicted_probabilities = model.predict_proba(features_test_scaled)

        metrics = compute_all_metrics(class_test, predicted_class, predicted_probabilities, class_label_ids)
        comparison_rows.append(metrics_to_row(model_name, metrics))

        model_filename = model_name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, args.output_dir / model_filename)
        print(f"Trained + saved {model_name} -> {args.output_dir / model_filename}")

    comparison_table = pd.DataFrame(comparison_rows).set_index("Model").round(4)
    print("\n=== Model comparison (test split, n={}) ===".format(len(class_test)))
    print(comparison_table.to_string())


if __name__ == "__main__":
    main()
