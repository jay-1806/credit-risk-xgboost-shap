# Credit Risk Prediction — XGBoost + SHAP

Predict credit risk on LendingClub loan data with an end-to-end ML pipeline (feature engineering → model training → explainability) and a Streamlit dashboard for interactive insights.

## Highlights
- **Model:** XGBoost classifier with Scikit-learn `Pipeline` + hyperparameter tuning.
- **Performance:** ROC-AUC ≈ **0.92** on a held-out test split.
- **Explainability:** SHAP global (summary) and local (force) explanations for auditability.
- **App:** Streamlit UI to explore risk by borrower segments and top SHAP features.

## Tech Stack
Python, XGBoost, scikit-learn, SHAP, Pandas, NumPy, Matplotlib/Plotly, Streamlit, Docker

## Dataset
- LendingClub public loan data (documented preprocessing: cleaning, feature typing, leakage checks).
- Train/val/test split with stratification; metrics: ROC-AUC, PR-AUC, calibration.

## Project Structure
credit-risk-xgboost-shap/
├─ app/ # Streamlit dashboard
│ ├─ Home.py
│ └─ pages/
│ ├─ 1_SHAP_Explorer.py
│ └─ 2_Segments.py
├─ data/
│ ├─ raw/ # (paths only; large files excluded)
│ └─ processed/
├─ src/
│ ├─ features/ # feature engineering, encoders
│ │ └─ build_features.py
│ ├─ models/
│ │ ├─ evaluate.py
│ ├─ explain/
│ │ └─ plots.py
│ └─ io.py
├─ configs/
│ ├─ params.yaml # model & preprocessing params
├─ tests/
│ └─ test_models.py
├─ requirements.txt
└─ README.md
