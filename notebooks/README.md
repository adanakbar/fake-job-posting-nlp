# 📒 Jupyter Notebooks — Project Workflow

This folder contains the key Jupyter notebooks for the **Fake Job Posting Detection using NLP** project. Each notebook documents a specific phase of the pipeline — from raw data preprocessing to final model training and evaluation.

---

## Notebooks Overview

### 1. `data_preprocessing.ipynb`
 *"Turning raw job listings into model-ready inputs."*

This notebook focuses on transforming the raw dataset into a structured, clean, and numerical format suitable for machine learning models.

#### Key Steps:
- Data loading and initial exploration
- Handling missing values
- Text column analysis (e.g., `description`, `requirements`)
- Feature engineering using:
  - TF-IDF Vectorization for textual features
  - Label Encoding / One-Hot Encoding for categorical features
  - Scaling of numeric columns (if necessary)

#### Outputs Saved:
- `X_final.npz` — Sparse matrix of processed features  
- `y_target.csv` — Target variable  
- `tfidf_vectorizer.pkl` — Trained TF-IDF vectorizer  
- `numeric_feature_names.csv` — Names of numerical features  

---

### 2. `model_training_evaluation.ipynb`
*"Training the brain to spot fake jobs."*

This notebook is responsible for training, validating, and saving the machine learning model.

#### Key Steps:
- Loading preprocessed data and vectorizer
- Splitting data into train/test sets
- Training an XGBoost Classifier
- Model evaluation using:
  - Accuracy, Precision, Recall, F1-score
  - ROC-AUC score
  - Confusion matrix and plots

#### Output Saved:
- `fraud_detection_xgb_model.pkl` — Final trained model for deployment

---

## Purpose of this Folder

This folder acts as a transparent and reproducible log of the entire model development workflow. It is structured to help:
- Understand the step-by-step data preparation and modeling process
- Collaborate effectively in a data science team
- Document experiments and pipeline decisions for future reference

---
**Note:** These notebooks are for experimentation and development. Production code is organized separately in the `app/` and `data/` folders.

