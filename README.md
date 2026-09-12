# 👨‍💼 Employee Attrition Prediction

A medium-level Machine Learning project that predicts whether an employee is likely to leave a company.

## Features

- Data generation and preprocessing
- Categorical feature encoding
- Random Forest classification
- Train/test split with stratification
- Accuracy evaluation
- Prediction probability
- Interactive Streamlit dashboard
- Basic exploratory visualization

## ML Pipeline

```text
Employee Data
      ↓
Preprocessing
      ↓
One-Hot Encoding
      ↓
Train/Test Split
      ↓
Random Forest
      ↓
Evaluation
      ↓
Prediction Dashboard
```

## Input Features

- Age
- Monthly income
- Years at company
- Overtime
- Job satisfaction
- Environment satisfaction
- Job level
- Distance from home
- Department
- Business travel

## Why Random Forest?

Random Forest combines many decision trees and can model non-linear relationships. It also handles mixed feature types well after categorical encoding and is a strong baseline for tabular classification.

## Run

```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Important

This version uses a **synthetically generated dataset** so the project is self-contained and can run immediately. For a production/research version, replace it with a real anonymized HR dataset and perform stronger validation, fairness checks, calibration, and privacy review.

## Future Improvements

- Add SHAP explainability
- Compare Logistic Regression, XGBoost and Random Forest
- Add ROC-AUC, precision, recall and F1 score
- Add cross-validation
- Deploy the application
- Add model monitoring
