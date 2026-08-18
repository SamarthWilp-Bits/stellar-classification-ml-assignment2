# Stellar Classification — SDSS17

## Problem Statement

Astronomical surveys like the Sloan Digital Sky Survey (SDSS) capture millions of
sky observations that need to be sorted into object types before any further science
can happen. This project builds a supervised classifier that takes photometric and
spectroscopic measurements of a single SDSS observation and predicts whether it is a
`GALAXY`, `STAR`, or `QSO` (quasar) — a 3-class classification problem. Six classifiers
are trained on the same preprocessed data, evaluated on an identical held-out split,
and served through a Streamlit app that loads the saved models (no retraining at app
startup).

## GitHub Repository Link

https://github.com/SamarthWilp-Bits/stellar-classification-ml-assignment2

## Live Streamlit App

https://stellar-classification-ml-assignment2-27nwznt4kkvhbjeakrvjtl.streamlit.app

## Dataset

- **Source**: [Stellar Classification Dataset - SDSS17](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17) (Kaggle, fedesoriano)
- **Size**: 100,000 rows, 17 raw feature columns + `class` target
- **Target**: `class` ∈ {`GALAXY`, `STAR`, `QSO`} — multi-class, imbalanced
  (≈45% GALAXY, ≈16% STAR, ≈14% QSO)

### Feature selection

Of the 17 raw columns, three are dropped before training:

| Column | Why dropped |
|---|---|
| `obj_ID`, `spec_obj_ID` | Unique per-row catalog identifiers (like a primary key) — no astrophysical signal, and including them risks the model memorizing row identity. |
| `rerun_ID` | Constant across every row in this SDSS data release — zero variance, contributes nothing. |

The remaining **14 features** (`alpha`, `delta`, `u`, `g`, `r`, `i`, `z`, `run_ID`,
`cam_col`, `field_ID`, `redshift`, `plate`, `MJD`, `fiber_ID`) are used for prediction —
comfortably clearing the assignment's 12-feature minimum even after the drops. Full
column definitions are in [`model/metrics_utils.py`](model/metrics_utils.py).

### Data cleaning

A handful of rows in this dataset have `u`/`g`/`z` values around `-9999` — a sentinel
for a failed photometric measurement, not a real magnitude (real SDSS magnitudes are
roughly 10–30). `model/train_models.py` drops any row where a magnitude column is
below `-100` before training.

## Repo structure

```
app.py                   Streamlit app — loads saved models, never retrains
requirements.txt         Pinned dependencies (see note on pickle compatibility below)
test_data.csv            Small held-out sample (500 rows) for the app's CSV upload demo
model/
  metrics_utils.py        Shared feature list + 6-metric computation (used by both training and app)
  train_models.py         Loads raw data, cleans it, trains + evaluates + saves all 6 models
  *.pkl                    Trained model artifacts, committed to the repo so the
                           Streamlit app can load them directly without retraining
data/
  (place star_classification.csv here — gitignored, not committed;
   the raw 100k-row file is not part of the repo)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Use this exact venv for both training and app development — see the pickle
compatibility note in `requirements.txt`.

## Training

1. Download `star_classification.csv` from the Kaggle link above and place it at
   `data/star_classification.csv` (this file is gitignored — 100k rows is too large
   and unnecessary to commit; only the small held-out sample gets committed).
2. Run:
   ```bash
   python model/train_models.py
   ```
   This prints a comparison table (all 6 models × all 6 metrics), writes the 6 `.pkl`
   models plus `feature_scaler.pkl` and `class_encoder.pkl` into `model/`, and writes
   `test_data.csv` at the repo root.

### Models trained

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. **Gaussian** Naive Bayes — chosen over Multinomial because all 14 features are
   continuous real-valued measurements (angles, magnitudes, redshift), and
   `delta` (declination) can be negative, which MultinomialNB can't accept at all
   (it requires non-negative, count-like data). GaussianNB models each feature as
   normally distributed per class, which fits continuous physical measurements.
5. Random Forest
6. Gradient Boosting — the assignment text states *"All the 6 ML models have to be
   implemented"* but explicitly names only 5 (Logistic Regression, Decision Tree,
   KNN, Naive Bayes, Random Forest). Gradient Boosting is included here as the
   candidate 6th model to satisfy the literal "6 models" wording; if your evaluator
   confirms only the 5 named models are required, delete the `"Gradient Boosting"`
   entry from `build_models()` in `model/train_models.py` — everything else
   (training loop, comparison table, app dropdown) adapts automatically since it
   just iterates over that dict.

### Metrics and averaging strategy

Accuracy, AUC, Precision, Recall, F1, MCC — computed identically in training and in
the app via `model/metrics_utils.py`.

- **AUC** uses `roc_auc_score(..., multi_class='ovr', average='macro')`.
- **Precision / Recall / F1** use `average='macro'`.
- **Why macro, not weighted**: the classes are imbalanced (GALAXY is ~3x QSO or
  STAR). Weighted averaging would let GALAXY performance dominate the score; macro
  scores each class equally, so a model that's lazy about telling the two rarer,
  physically more similar classes (STAR vs. QSO) apart doesn't get to hide behind
  its GALAXY accuracy.
- **MCC** (`matthews_corrcoef`) needs no averaging parameter — it's a single balanced
  coefficient for multiclass by construction, which is why it's included alongside
  accuracy as an imbalance-robust check.

## Running the app locally

```bash
streamlit run app.py
```

Upload `test_data.csv` (or any CSV with the same 14 feature columns + `class`
column) and pick a model from the dropdown. Malformed uploads (missing columns,
non-numeric values, unknown class labels) are caught and shown as an error message
instead of crashing the app.

## Deploying to Streamlit Cloud

See the checklist at the bottom of this doc for the manual steps (push to GitHub,
connect the repo, set the Python version). The short version: point Streamlit Cloud
at `app.py`, and make sure the `.pkl` files under `model/` are committed (Streamlit
Cloud only has what's in the repo — it doesn't run `train_models.py` for you).

## Model comparison

Test split: n=20,000 (stratified 20% of the cleaned 99,999-row dataset).

```
                      Accuracy     AUC  Precision  Recall      F1     MCC
Model
Logistic Regression     0.9599  0.9880     0.9568  0.9520  0.9538  0.9290
Decision Tree           0.9671  0.9710     0.9619  0.9622  0.9621  0.9417
K-Nearest Neighbors     0.9149  0.9618     0.9240  0.8843  0.9023  0.8471
Gaussian Naive Bayes    0.9017  0.9714     0.8803  0.9168  0.8942  0.8362
Random Forest           0.9792  0.9950     0.9789  0.9732  0.9759  0.9631
Gradient Boosting       0.9776  0.9951     0.9789  0.9690  0.9737  0.9602
```

### Reproducing these numbers via the app

The table above is computed on the full 20,000-row test split at training time. The
Streamlit app (both local and the deployed version) instead scores whatever CSV you
upload — by default that's `test_data.csv`, a smaller 500-row stratified sample of
that same test split, included so uploads stay fast on Streamlit Cloud's free tier.
Because it's a smaller sample, the numbers you see in the app will be close to, but
not identical to, the table above — this is expected sampling variance, not an
inconsistency. For reference, here's what the deployed app actually reports on
`test_data.csv` (verified by running all 6 models against the live app):

```
                      Accuracy     AUC  Precision  Recall      F1     MCC
Model
Logistic Regression     0.9440  0.9843     0.9366  0.9399  0.9371  0.9018
Decision Tree           0.9560  0.9642     0.9459  0.9538  0.9498  0.9226
K-Nearest Neighbors     0.8840  0.9508     0.8852  0.8480  0.8649  0.7908
Gaussian Naive Bayes    0.8780  0.9630     0.8553  0.8942  0.8697  0.7978
Random Forest           0.9760  0.9952     0.9770  0.9674  0.9720  0.9574
Gradient Boosting       0.9740  0.9956     0.9758  0.9639  0.9695  0.9538
```

The ranking and the Overall Winner conclusion below are unchanged on this smaller
sample: Random Forest still leads on Accuracy/F1/MCC, and Gradient Boosting still
edges it out on AUC alone by a similarly thin margin.

## Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong baseline (95.99% accuracy, 0.988 AUC) despite being a linear model — the three classes are largely linearly separable once features are scaled, particularly because `redshift` alone carries a lot of signal (QSOs and distant galaxies have systematically higher redshift than stars). Precision/recall are balanced (~0.95 each), so it isn't biased toward the majority GALAXY class. |
| Decision Tree | Outperforms Logistic Regression on accuracy/F1/MCC (0.9671/0.9621/0.9417) by capturing non-linear feature interactions, but has the lowest AUC among the top 4 models (0.971) — a single unpruned tree gives fairly extreme, poorly-calibrated class probabilities, which hurts ranking-based metrics like AUC even when its hard predictions are accurate. |
| K-Nearest Neighbors | Second-weakest model on accuracy/precision/F1/MCC (91.5% accuracy, 90.2% F1, 0.847 MCC) — beats Naive Bayes on these four metrics, but has the lowest recall of all six models (0.884). With 14 features, distance-based neighbor lookup starts to suffer from the curse of dimensionality, and STAR/QSO — the two minority, spectroscopically similar classes — are the most likely to get missed (false negatives), which is what recall penalizes. |
| Gaussian Naive Bayes | **Weakest model overall** — lowest accuracy (90.2%), precision (0.880), F1 (0.894), and MCC (0.836) of all six models. Its core independence assumption (each feature is normally distributed and uncorrelated given the class) doesn't hold well here, since the five magnitude columns (`u,g,r,i,z`) are strongly correlated with each other. Interestingly it still beats KNN on AUC (0.971 vs 0.962) and recall (0.917 vs 0.884) — its probabilistic outputs rank classes reasonably well even when its hard classifications are the least accurate. It's also the fastest to train by a wide margin, which is its main practical advantage. |
| Random Forest | **Best performing model overall** — highest accuracy (97.92%), AUC (0.9950), F1 (0.9759), and MCC (0.9631). Bagging many decision trees fixes the single-tree's calibration weakness (much higher AUC than the standalone Decision Tree) while keeping its ability to model non-linear class boundaries. |
| Gaussian Naive Bayes vs Gradient Boosting | Gradient Boosting is essentially tied with Random Forest (97.76% accuracy, 0.9951 AUC — marginally the single highest AUC of all six models) but very slightly behind on F1/MCC, likely because boosting optimizes a different loss trajectory and Random Forest's bagging generalizes marginally better on this particular stratified split. |
| **Overall Winner for this dataset** | **Random Forest.** Selection criterion: primarily macro-F1 and MCC (chosen over raw accuracy because the classes are imbalanced ~45/16/14%, so a metric that scores each class equally is more trustworthy), with AUC as a tie-breaker for ranking quality. Random Forest leads on F1 (0.9759) and MCC (0.9631), and is within 0.0001 of Gradient Boosting's AUC — the single best-performing model without any hyperparameter tuning. |

---

## Project status

- [x] Dataset downloaded and trained on (99,999 clean rows, 14 features)
- [x] All 6 models trained, evaluated on an identical held-out split, comparison
      table above reflects the actual run
- [x] App verified locally (`streamlit run app.py`) — upload, dropdown, metrics,
      confusion matrix, and classification report all confirmed working
- [x] Pushed to GitHub, all required files present in the repo
- [x] Deployed to Streamlit Community Cloud, live app verified end-to-end with the
      same results as the local run
- [ ] BITS Virtual Lab screenshot pending
