import streamlit as st
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix

# ========== App Configuration ==========
st.set_page_config(page_title="Job Fraud Detector", page_icon="🔍", layout="wide")

# ========== Load Model & Assets ==========
model = joblib.load("fraud_detection_xgb_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
numeric_cols = pd.read_csv("numeric_feature_names.csv", header=None)[0].tolist()
default_numeric = np.zeros((1, len(numeric_cols)))


# ========== App Interface ==========
def run_app():
    st.title("Job Fraud Detection")
    st.markdown(
        "This tool helps assess whether a job description might be fraudulent based on textual patterns and metadata."
    )
    st.markdown("")

    job_text = st.text_area(
        "Job Description",
        height=200,
        placeholder="Paste the job posting content here...",
    )

    if st.button("Run Prediction"):
        if not job_text.strip():
            st.warning("Please provide a job description.")
            return

        with st.spinner("Processing..."):
            # Text transformation
            tfidf_transformed = vectorizer.transform([job_text])
            numeric_dummy_sparse = csr_matrix(default_numeric)

            # Combine text + numeric
            final_input = hstack([tfidf_transformed, numeric_dummy_sparse])
            current_features = final_input.shape[1]
            expected_features = model.n_features_in_

            # Match shape
            if current_features > expected_features:
                final_input = final_input[:, :expected_features]
            elif current_features < expected_features:
                padding = csr_matrix((1, expected_features - current_features))
                final_input = hstack([final_input, padding])

            # Predict
            proba = model.predict_proba(final_input)[0][1]
            prediction = model.predict(final_input)[0]

        st.markdown("---")
        if prediction == 1:
            st.markdown(f"### ❌ This job posting may be fraudulent.")
            st.markdown(f"**Model confidence:** {proba:.2f}")
        else:
            st.markdown(f"### ✅ This job posting appears legitimate.")
            st.markdown(f"**Model confidence:** {1 - proba:.2f}")

    with st.expander("What does this tool do?"):
        st.markdown(
            """
        This tool analyzes job postings using a machine learning model trained on labeled examples of real and fraudulent job descriptions.
        
        It uses:
        - Text vectorization (TF-IDF)
        - Known patterns from fraudulent listings
        - Statistical modeling (XGBoost Classifier)
        
        The prediction is based solely on text structure and feature patterns—not on company names or external reputation.
        """
        )


# ========== Run ==========
if __name__ == "__main__":
    run_app()
