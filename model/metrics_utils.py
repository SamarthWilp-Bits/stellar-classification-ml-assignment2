"""
Shared constants and metric computation for the stellar classification project.

Imported by both model/train_models.py (training-time evaluation) and app.py
(evaluation of user-uploaded test CSVs), so the numbers reported in the
training console output and in the Streamlit app are computed by the exact
same code path and can never silently drift apart.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

# --- Dataset schema -----------------------------------------------------
# Columns present in the raw SDSS17 "star_classification.csv" export.
TARGET_COLUMN = "class"

# obj_ID and spec_obj_ID are unique per-row catalog identifiers (like a
# primary key) — they carry no astrophysical signal and including them would
# just let models memorize row identity instead of learning from the
# photometry. rerun_ID is constant across every row in this SDSS data
# release (always the same processing rerun), so it has zero variance and
# contributes nothing. All three are dropped before training.
ID_COLUMNS_TO_DROP = ["obj_ID", "spec_obj_ID", "rerun_ID"]

# The 14 columns actually used to predict `class`. This clears the
# assignment's 12-feature minimum even after dropping the three columns
# above.
FEATURE_COLUMNS = [
    "alpha",     # right ascension (deg)
    "delta",     # declination (deg)
    "u",         # ultraviolet filter magnitude
    "g",         # green filter magnitude
    "r",         # red filter magnitude
    "i",         # near-infrared filter magnitude
    "z",         # infrared filter magnitude
    "run_ID",    # scan run number
    "cam_col",   # camera column (1-6) within the run
    "field_ID",  # field number within the run
    "redshift",  # spectroscopic redshift
    "plate",     # SDSS plate ID
    "MJD",       # modified Julian date of observation
    "fiber_ID",  # spectroscopic fiber ID
]

# Real SDSS photometric magnitudes fall roughly in the 10-30 range. This
# dataset has a small number of rows where the u/g/z pipeline failed and the
# sentinel value -9999 was written instead of a real magnitude. We treat
# anything below this threshold as a bad/missing measurement and drop the
# row rather than let it distort the scaler and every downstream model.
BAD_MAGNITUDE_THRESHOLD = -100.0

CLASS_LABELS_ORDER = ["GALAXY", "QSO", "STAR"]  # alphabetical; overwritten by the fitted LabelEncoder at train time

# --- Metrics --------------------------------------------------------------
# All averaged metrics use 'macro' averaging: this dataset is imbalanced
# (GALAXY ~45% of rows, STAR ~16%, QSO ~14%), and macro-averaging scores each
# class equally instead of letting the majority GALAXY class dominate the
# number. That matters here because a model that is lazy about telling STAR
# and QSO apart (the two minority, and physically more similar, classes)
# should NOT still look great just because it nails the easy majority class.
AVERAGING_STRATEGY = "macro"


def compute_all_metrics(true_labels, predicted_labels, predicted_probabilities, class_labels):
    """
    Compute the 6 required metrics for one model's predictions.

    true_labels / predicted_labels : 1D arrays of integer-encoded class labels
    predicted_probabilities        : 2D array (n_samples, n_classes) from predict_proba,
                                      with columns ordered to match `class_labels`
    class_labels                   : sorted array of the integer-encoded class labels
                                      the model was trained on (e.g. [0, 1, 2])

    Returns a dict with keys: Accuracy, AUC, Precision, Recall, F1, MCC.
    AUC is np.nan if it can't be computed (e.g. the uploaded test set doesn't
    contain all classes) — callers must handle that rather than crash.
    """
    metrics = {
        "Accuracy": accuracy_score(true_labels, predicted_labels),
        "Precision": precision_score(true_labels, predicted_labels, average=AVERAGING_STRATEGY, zero_division=0),
        "Recall": recall_score(true_labels, predicted_labels, average=AVERAGING_STRATEGY, zero_division=0),
        "F1": f1_score(true_labels, predicted_labels, average=AVERAGING_STRATEGY, zero_division=0),
        # MCC is inherently a single balanced coefficient for multiclass —
        # it takes no averaging parameter, which is part of why it's a good
        # complement to accuracy on an imbalanced dataset.
        "MCC": matthews_corrcoef(true_labels, predicted_labels),
    }

    try:
        metrics["AUC"] = roc_auc_score(
            true_labels,
            predicted_probabilities,
            multi_class="ovr",
            average=AVERAGING_STRATEGY,
            labels=class_labels,
        )
    except ValueError:
        # Most commonly: the test set is missing one of the classes the
        # model was trained on, which roc_auc_score's ovr mode can't score.
        metrics["AUC"] = np.nan

    return metrics


def metrics_to_row(model_name, metrics_dict):
    """Flatten one model's metrics dict into a single-row dict for a comparison table."""
    row = {"Model": model_name}
    row.update({k: metrics_dict[k] for k in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]})
    return row
