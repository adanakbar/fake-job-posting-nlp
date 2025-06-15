# Data Artifacts — Preprocessed Features & Model Assets

This folder stores all the **preprocessed data files**, **trained model**, and **vectorizer** necessary to run both the training notebooks and the Streamlit app.

These files are generated during the preprocessing and training phases and are reused during inference in the app.

---

## Files Overview

### `X_final.npz`
- A sparse matrix of transformed feature vectors combining TF-IDF text features and scaled/encoded numerical features.
- Input features used for training and prediction.

### `y_target.csv`
- The target variable (`fraudulent`) corresponding to each row in `X_final.npz`.
- Used for supervised model training and evaluation.

### `tfidf_vectorizer.pkl`
- The fitted **TF-IDF Vectorizer** used to transform job descriptions into numerical feature vectors.
- Ensures consistency between training and inference.

### `numeric_feature_names.csv`
- List of the column names of additional numeric or categorical features included alongside TF-IDF vectors.
- Used to reconstruct feature space at inference time.

### `fraud_detection_xgb_model.pkl`
- The trained **XGBoost Classifier** model for predicting whether a job posting is fake or legitimate.
- Loaded directly in the Streamlit app for real-time predictions.

---

## Purpose of the `data/` Folder

This directory acts as a central storage hub for all important intermediate and final artifacts. It ensures that:
- The model and app can work independently of the notebooks
- Preprocessing and modeling steps don't need to be re-run unnecessarily
- Deployment pipelines can access standardized, saved files

> ⚠️ **Note:** These files may be large and can be excluded from GitHub using `.gitignore` if needed. For production, consider using cloud storage or versioned ML artifact tracking tools like DVC or MLflow.

