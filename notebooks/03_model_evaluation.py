# -*- coding: utf-8 -*-
"""
03_model_evaluation.py
----------------------
This script handles:
    - Loading all trained models & test data
    - Evaluating models using accuracy, precision, recall, and f1-score
    - Plotting Confusion Matrix and ROC Curve
    - Saving evaluation report for all models
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.model_selection import train_test_split


def plot_confusion_matrix(y_true, y_pred, model_name: str, reports_dir: Path):
    """Generate and save confusion matrix as PNG"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    save_path = reports_dir / f"{model_name}_confusion_matrix.png"
    plt.savefig(save_path)
    plt.close()
    print(f"[SUCCESS] Saved confusion matrix → {save_path}")


def plot_roc_curve(y_true, y_pred_prob, model_name: str, reports_dir: Path):
    """Generate and save ROC curve as PNG"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    save_path = reports_dir / f"{model_name}_roc_curve.png"
    plt.savefig(save_path)
    plt.close()
    print(f"[SUCCESS] Saved ROC curve → {save_path}")


def evaluate_models(models_dir: Path, X_test, y_test, reports_dir: Path):
    """Load all models from models_dir, evaluate and save results"""
    reports = {}

    for model_file in models_dir.glob("*.pkl"):
        model_name = model_file.stem
        try:
            model = joblib.load(model_file)
            print(f"[INFO] Loaded model: {model_name}")

            y_pred = model.predict(X_test)

            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            reports[model_name] = report["weighted avg"]
            print(f"[SUCCESS] Evaluation done for: {model_name}")

            # Confusion Matrix
            plot_confusion_matrix(y_test, y_pred, model_name, reports_dir)

            # ROC Curve (only if model supports predict_proba)
            if hasattr(model, "predict_proba"):
                y_pred_prob = model.predict_proba(X_test)[:, 1]
                plot_roc_curve(y_test, y_pred_prob, model_name, reports_dir)
            else:
                print(f"[WARNING] {model_name} does not support predict_proba → Skipping ROC")

        except Exception as e:
            print(f"[ERROR] Failed to evaluate {model_name} → {e}")

    # Save combined report
    if reports:
        report_df = pd.DataFrame(reports).transpose()
        report_path = reports_dir / "evaluation_report.csv"
        report_df.to_csv(report_path, index=True)
        print(f"[SUCCESS] Combined evaluation report saved → {report_path}")
    else:
        print("[WARNING] No reports generated, empty reports dictionary")


def main():
    """Main execution function"""
    try:
        # Paths setup
        HERE = Path(__file__).resolve().parent
        ROOT = HERE.parent
        DATA_DIR = ROOT / "data"
        MODELS_DIR = ROOT / "models"
        REPORTS_DIR = ROOT / "reports"
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # Load cleaned dataset
        clean_path = DATA_DIR / "cleaned_telco.csv"
        if not clean_path.exists():
            raise FileNotFoundError(f"Cleaned dataset not found at {clean_path}")

        df = pd.read_csv(clean_path)

        if "Churn" not in df.columns:
            raise KeyError("Target column 'Churn' not found in dataset")

        # Train/test split
        X = df.drop(columns=["Churn"])
        y = df["Churn"]

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Evaluate all models
        evaluate_models(MODELS_DIR, X_test, y_test, REPORTS_DIR)

    except Exception as e:
        print(f"[ERROR] Model evaluation failed → {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
