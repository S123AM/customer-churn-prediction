# -*- coding: utf-8 -*-
"""
04_inference.py
---------------
Handles model inference:
    - Selects best model from evaluation_report.csv
    - Loads the best model
    - Predicts churn for new/unseen customer data
"""

import pandas as pd
from pathlib import Path
import joblib
import sys
import logging

# -------------------- LOGGER --------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

def get_best_model(report_path: Path) -> str:
    """Select best model based on f1-score (weighted avg)."""
    if not report_path.exists():
        logging.error(f"Evaluation report not found at {report_path}")
        sys.exit(1)

    report = pd.read_csv(report_path, index_col=0)

    model_scores = {
        model: report.loc[model, "f1-score"]
        for model in ["logistic_regression", "random_forest", "xgboost"]
        if model in report.index
    }

    if not model_scores:
        logging.error("No models found in evaluation report")
        sys.exit(1)

    best_model = max(model_scores, key=model_scores.get)
    return best_model


def predict_new(data: pd.DataFrame, model_name: str):
    """Predict churn + probability for new customers."""
    ROOT = Path(__file__).resolve().parent.parent
    model_path = ROOT / "models" / f"{model_name}.pkl"

    if not model_path.exists():
        logging.error(f"Model not found at {model_path}")
        return None

    model = joblib.load(model_path)

    # Prediction + probability
    preds = model.predict(data)
    probas = model.predict_proba(data)[:, 1]
    return preds, probas


def main():
    try:
        ROOT = Path(__file__).resolve().parent.parent
        report_path = ROOT / "reports" / "evaluation_report.csv"

        # -------------------- BEST MODEL --------------------
        best_model = get_best_model(report_path)
        logging.info(f"Best model selected: {best_model}")

        # -------------------- INPUT DATA --------------------
        # Example new customers (could later read from CSV/JSON)
        example_data = {
            "gender": ["Male", "Female"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [5, 20],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "Yes"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "Yes"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "Yes"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [70.5, 99.9],
            "TotalCharges": [350.5, 2000.0],
        }
        new_customers = pd.DataFrame(example_data)

        # -------------------- PREDICT --------------------
        preds, probas = predict_new(new_customers, best_model)
        if preds is not None:
            logging.info(f"Predictions using {best_model}:")
            for i, (p, prob) in enumerate(zip(preds, probas)):
                label = "Churn" if p == 1 else "Stay"
                print(f"  Customer {i+1}: {label} (prob: {prob:.2f})")

    except Exception as e:
        logging.error(f"Inference failed → {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
