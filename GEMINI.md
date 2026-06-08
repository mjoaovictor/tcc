# Stroke Prediction Project

## Overview
This project aims to predict the likelihood of a patient having a stroke based on clinical features such as age, hypertension, heart disease, smoking status, and BMI. It implements a complete Machine Learning lifecycle, ranging from exploratory data analysis (EDA) to hyperparameter tuning and standardized model evaluation.

## Tech Stack
- **Language:** Python 3.12+
- **Dependency Management:** `uv` (preferred), `pip`
- **Data Science:** `pandas`, `numpy`, `scipy`
- **Machine Learning:** `scikit-learn`, `xgboost`, `imbalanced-learn`
- **Visualization:** `matplotlib`, `seaborn`
- **Linting & Formatting:** `ruff`

## Architecture
- `data/`: contains the dataset in raw (`stroke.csv`) and processed versions.
- `notebooks/`: structured experimental lifecycle.
    - `01_eda.ipynb`: exploratory data analysis and feature distribution.
    - `02_modeling.ipynb`: baseline model development and feature engineering.
    - `03_tuning.ipynb`: hyperparameter optimization using `RandomizedSearchCV`.
    - `04_experiments.ipynb`: final performance comparison across models.
- `src/`: modularized core logic.
    - `preprocessing.py`: handles data cleaning, string normalization, and provides a robust Scikit-Learn `Pipeline` builder with support for imbalanced data samplers (e.g., `SMOTETomek`).
    - `metrics.py`: provides standardized evaluation metrics including PR-AUC, ROC-AUC, and Brier Score, along with threshold optimization logic.

## Key Commands
- **Environment Setup:** `uv venv` followed by `uv pip install -r requirements.txt`.
- **Linting:** `ruff check .` to verify adherence to standards.
- **Formatting:** `ruff format .` to apply code style conventions.
- **Notebooks:** execute via JupyterLab or Jupyter Notebook.

## Development Conventions
- **Language:** 100% English for all code, documentation, and comments.
- **Tone:** professional, execution-driven, and minimalist.
- **Comments:** all comments MUST be in lowercase, except for technical terms (e.g., # initialize the CLI via API).
- **Standards:**
    - strict adherence to clean code and surgical updates.
    - line length limit of 79 characters.
    - double quotes for strings.
    - 4-space indentation.
- **Testing:** validation is mandatory for all changes; ensure metrics are tracked consistently using `src/metrics.py`.
