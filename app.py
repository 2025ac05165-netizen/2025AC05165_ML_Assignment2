"""
app.py
------
Streamlit app for BITS Pilani ML Assignment 2.

Features:
  a. Dataset upload option (CSV) - upload test data
  b. Model selection dropdown
  c. Display of evaluation metrics
  d. Confusion matrix / classification report

Dataset: Breast Cancer Wisconsin (Diagnostic) - binary classification
Target column expected in the uploaded CSV: "target" (0 = malignant, 1 = benign)
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="ML Assignment 2 - Classification Models Demo",
    page_icon="🔬",
    layout="wide",
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
    models = {
        name: joblib.load(os.path.join(MODEL_DIR, fname))
        for name, fname in MODEL_FILES.items()
    }
    return scaler, feature_names, models


scaler, feature_names, models = load_artifacts()

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("🔬 Breast Cancer Classification — Model Comparison App")
st.markdown(
    """
    This app demonstrates **5 classification models** trained on the
    [Breast Cancer Wisconsin (Diagnostic) dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset)
    (569 instances, 30 features, binary classification: malignant vs. benign).

    Upload the provided `test_data.csv`, pick a model, and view its evaluation metrics,
    confusion matrix, and classification report.
    """
)

# ---------------------------------------------------------------------
# a. Dataset upload
# ---------------------------------------------------------------------
st.header("1. Upload Test Data (CSV)")
uploaded_file = st.file_uploader(
    "Upload test_data.csv (must include a 'target' column)", type=["csv"]
)

if uploaded_file is None:
    st.info("👆 Upload the `test_data.csv` file from the GitHub repo to get started.")
    st.stop()

df = pd.read_csv(uploaded_file)

if "target" not in df.columns:
    st.error("The uploaded CSV must contain a 'target' column with the true labels.")
    st.stop()

missing_cols = [c for c in feature_names if c not in df.columns]
if missing_cols:
    st.error(f"Uploaded CSV is missing required feature columns: {missing_cols}")
    st.stop()

st.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
with st.expander("Preview uploaded data"):
    st.dataframe(df.head())

X = df[feature_names]
y_true = df["target"]
X_scaled = scaler.transform(X)

# ---------------------------------------------------------------------
# b. Model selection dropdown
# ---------------------------------------------------------------------
st.header("2. Select a Model")
model_name = st.selectbox("Choose a classification model:", list(models.keys()))
model = models[model_name]

y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)[:, 1]

# ---------------------------------------------------------------------
# c. Evaluation metrics
# ---------------------------------------------------------------------
st.header("3. Evaluation Metrics")

acc = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_proba)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Accuracy", f"{acc:.4f}")
col2.metric("AUC", f"{auc:.4f}")
col3.metric("Precision", f"{prec:.4f}")
col4.metric("Recall", f"{rec:.4f}")
col5.metric("F1 Score", f"{f1:.4f}")
col6.metric("MCC", f"{mcc:.4f}")

# ---------------------------------------------------------------------
# All-models comparison table (bonus, uses the uploaded data)
# ---------------------------------------------------------------------
st.subheader("Compare All Models on This Data")
comparison_rows = []
for name, m in models.items():
    p = m.predict(X_scaled)
    pr = m.predict_proba(X_scaled)[:, 1]
    comparison_rows.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_true, p), 4),
        "AUC": round(roc_auc_score(y_true, pr), 4),
        "Precision": round(precision_score(y_true, p), 4),
        "Recall": round(recall_score(y_true, p), 4),
        "F1": round(f1_score(y_true, p), 4),
        "MCC": round(matthews_corrcoef(y_true, p), 4),
    })
comparison_df = pd.DataFrame(comparison_rows).set_index("Model")
st.dataframe(comparison_df.style.highlight_max(axis=0, color="lightgreen"))

# ---------------------------------------------------------------------
# d. Confusion matrix / classification report
# ---------------------------------------------------------------------
st.header("4. Confusion Matrix & Classification Report")

left, right = st.columns(2)

with left:
    st.subheader(f"Confusion Matrix — {model_name}")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Malignant (0)", "Benign (1)"],
        yticklabels=["Malignant (0)", "Benign (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.subheader(f"Classification Report — {model_name}")
    report = classification_report(y_true, y_pred, target_names=["Malignant", "Benign"])
    st.text(report)

st.caption(
    "Built for BITS Pilani WILP M.Tech (AIML/DSE) — Machine Learning, Assignment 2."
)
