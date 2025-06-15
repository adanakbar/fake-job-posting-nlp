# Streamlit App — Fraud Job Posting Detector

This folder contains the **Streamlit-based web application** for the Fake Job Posting Detection project. The app provides a user-friendly interface that allows users to input job posting text and receive a prediction on whether the job is likely to be **fraudulent or legitimate**.

---

## File Overview

### `fraud_app.py`
*"Real-time job listing fraud detection in a browser-friendly UI."*

This script runs the entire web app using Streamlit. It includes:
- A title and interactive input form
- Preprocessing of user-entered job descriptions
- Loading of the pre-trained:
  - TF-IDF vectorizer (`tfidf_vectorizer.pkl`)
  - XGBoost classifier model (`fraud_detection_xgb_model.pkl`)
- Real-time prediction and result display

---

## Features

- Simple interface for end users to paste job descriptions
- Uses machine learning to predict fraudulent postings
- Instant feedback with visual cues (fraud or not)
- Built using the Streamlit framework

---

## How to Run the App

Make sure the following files are in the root or `data/` folder:
- `fraud_detection_xgb_model.pkl`
- `tfidf_vectorizer.pkl`

### Run locally:
```bash
cd app
streamlit run fraud_app.py
