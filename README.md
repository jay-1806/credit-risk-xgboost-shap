# Credit Risk Prediction — XGBoost + SHAP
![python](https://img.shields.io/badge/Python-3.10+-blue)
![license](https://img.shields.io/badge/License-MIT-green)
![status](https://img.shields.io/badge/Status-WIP-orange)

Predict credit risk using an XGBoost pipeline with explainable AI (SHAP). Designed for finance teams that need both high performance **and** transparent, audit-friendly insights.

## Highlights
- **Model:** XGBoost classifier with robust preprocessing and hyperparameter tuning
- **Performance:** ROC-AUC ≈ **0.92** on LendingClub test data
- **Explainability:** SHAP summary/force plots to explain global & local predictions
- **App:** Streamlit dashboard to explore borrower segments and risk drivers

## Dataset
- **Source:** LendingClub public loan data (documented in repo)
- **Notes:** Includes cleaning, feature engineering (dates, ratios, categorical encodings)
- Ensure compliance with data usage and privacy requirements

## Project Structure
