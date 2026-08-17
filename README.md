# ML Assignment 2 — Classification Models

---

## a. Problem Statement

The goal of this assignment is to build, evaluate, and deploy multiple
classification models on a single dataset, and to expose the results through
an interactive Streamlit web application. Specifically, the task is to
predict whether a breast tumor is **malignant** or **benign** based on
diagnostic measurements derived from a digitized image of a fine needle
aspirate (FNA) of a breast mass. This is a **binary classification** problem
with real-world clinical relevance (early cancer diagnosis support).

## b. Dataset Description

- **Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** UCI Machine Learning Repository (also bundled with
  scikit-learn as `sklearn.datasets.load_breast_cancer`)
- **Instances:** 569 (≥ 500 required ✅)
- **Features:** 30 numeric features (≥ 12 required ✅) — computed from cell
  nuclei measurements, e.g. `mean radius`, `mean texture`, `mean perimeter`,
  `mean smoothness`, `worst concavity`, `worst symmetry`, etc. (10 base
  measurements × 3 summary statistics: mean, standard error, worst)
- **Target variable:** `target` — 0 = malignant, 1 = benign
- **Class balance:** 212 malignant, 357 benign
- **Train/test split:** 80% train (455 rows) / 20% test (114 rows),
  stratified by class

## c. GitHub Repository Link

> **https://github.com/2025ac05165-netizen/2025AC05165_ML_Assignment2.git**

Repository structure:
```
project-folder/
│-- app.py                 # Streamlit application
│-- requirements.txt       # Python dependencies
│-- README.md              # This file
│-- test_data.csv          # Test split used for evaluation (114 rows)
│-- model/
│   │-- train_models.py           # Trains all 5 models, saves artifacts
│   │-- logistic_regression.joblib
│   │-- decision_tree.joblib
│   │-- knn.joblib
│   │-- naive_bayes.joblib
│   │-- random_forest_ensemble.joblib
│   │-- scaler.joblib             # Fitted StandardScaler
│   │-- feature_names.joblib      # Ordered list of feature columns
│   │-- metrics_comparison.csv    # Metrics generated at training time
```

## d. Models Used

All 5 models were trained on the **same** 80/20 train-test split of the
Breast Cancer Wisconsin dataset, with features standardized using
`StandardScaler` (fit on the training set only).

### Comparison Table

| ML Model Name             | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|----------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression         | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree                | 0.9123   | 0.9157 | 0.9559    | 0.9028 | 0.9286 | 0.8174 |
| kNN                           | 0.9561   | 0.9788 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes                   | 0.9298   | 0.9868 | 0.9444    | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble)      | 0.9561   | 0.9932 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |

### Observations

| ML Model Name              | Observation about model performance |
|-----------------------------|--------------------------------------|
| Logistic Regression          | Best overall performer on every metric (Accuracy 0.9825, MCC 0.9623). The classes are close to linearly separable after scaling, so a linear decision boundary generalizes very well and doesn't overfit the small feature set. |
| Decision Tree                | Weakest model (Accuracy 0.9123, AUC 0.9157, lowest MCC). A single unpruned tree overfits the training data and is sensitive to small variations, leading to more misclassifications and a much lower AUC than the other models. |
| kNN                           | Strong performance (Accuracy 0.9561, F1 0.9655) after feature scaling, since kNN is distance-based and standardization prevents large-magnitude features (e.g. area) from dominating the distance metric. |
| Naive Bayes                   | Decent Accuracy (0.9298) but the highest AUC among the non-ensemble/non-LR models isn't matched by its Accuracy/F1 — the conditional-independence assumption doesn't perfectly hold for these correlated cell-measurement features, capping its precision/recall balance. |
| Random Forest (Ensemble)      | Matches kNN on Accuracy/F1 (0.9561 / 0.9655) but with a notably higher AUC (0.9932), confirming that averaging many decision trees fixes the overfitting problem seen in the single Decision Tree and produces much better-calibrated probability estimates. |
| **Overall Winner for this dataset** | **Logistic Regression** — it achieves the top score on every single metric (Accuracy, AUC, Precision, Recall, F1, MCC), suggesting the diagnostic features separate malignant vs. benign tumors in a nearly linear fashion once standardized. Random Forest is the strongest runner-up, offering the best AUC after Logistic Regression and far more robustness than a single Decision Tree. |

## Live Streamlit App

> **https://cdbsdcuqofvxeqejyqfaxp.streamlit.app/**

## App Features

- **Dataset upload (CSV):** upload `test_data.csv` to evaluate models on the
  held-out test split.
- **Model selection dropdown:** choose any of the 5 trained models.
- **Evaluation metrics display:** Accuracy, AUC, Precision, Recall, F1, MCC
  shown live for the selected model, plus a full comparison table across all
  5 models on the uploaded data.
- **Confusion matrix & classification report:** visual confusion matrix
  (heatmap) and per-class precision/recall/F1 report for the selected model.
