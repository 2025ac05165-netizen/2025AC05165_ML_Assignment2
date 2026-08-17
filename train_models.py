"""
train_models.py
----------------
Trains 5 classification models on the Breast Cancer Wisconsin (Diagnostic)
dataset, evaluates them with 6 metrics, and saves:
  - trained models (model/*.joblib)
  - the fitted StandardScaler (model/scaler.joblib)
  - the test split used for evaluation (test_data.csv, at project root)
  - a metrics comparison table (model/metrics_comparison.csv)

Dataset: Breast Cancer Wisconsin (Diagnostic)
  - Source: UCI Machine Learning Repository / built into scikit-learn
  - Instances: 569
  - Features: 30 (satisfies the >=12 feature requirement)
  - Task: Binary classification (malignant vs benign)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target  # 0 = malignant, 1 = benign
feature_names = list(X.columns)

print(f"Dataset shape: {X.shape[0]} instances, {X.shape[1]} features")
print(f"Class distribution:\n{y.value_counts()}")

# ---------------------------------------------------------------------
# 2. Train/test split
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Save the test split (features + target) as test_data.csv -> used by the
# Streamlit app and required as part of the GitHub submission.
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)
print(f"Saved test_data.csv with {test_df.shape[0]} rows")

# ---------------------------------------------------------------------
# 3. Scale features (kNN / Logistic Regression benefit from scaling)
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(HERE, "scaler.joblib"))
joblib.dump(feature_names, os.path.join(HERE, "feature_names.joblib"))

# ---------------------------------------------------------------------
# 4. Define models
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE
    ),
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)
    print(metrics)

    # Save model
    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, os.path.join(HERE, f"{fname}.joblib"))

# ---------------------------------------------------------------------
# 5. Save comparison table
# ---------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(HERE, "metrics_comparison.csv"), index=False)
print("\nComparison table:\n", results_df.to_string(index=False))
print("\nAll models and artifacts saved in model/")
