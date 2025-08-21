# -*- coding: utf-8 -*-
"""
evaluate.py
-----------
Utility functions for model evaluation:
    - Confusion Matrix plotting
    - ROC Curve plotting
    - Model evaluation (classification report, ROC-AUC)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)


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


def evaluate_single_model(model, model_name: str, X_test, y_test, reports_dir: Path):
    """
    Evaluate a single trained model:
        - Classification Report
        - Confusion Matrix
        - ROC Curve (if applicable)
        - ROC-AUC score
    Returns:
        dict: Weighted average classification metrics
    """
    # Predictions
    y_pred = model.predict(X_test)

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    weighted_avg = report["weighted avg"]

    # Print metrics
    print(f"\n[INFO] Evaluation results for: {model_name}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, model_name, reports_dir)

    # ROC Curve & ROC-AUC (if model supports predict_proba)
    if hasattr(model, "predict_proba"):
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, y_pred_prob, model_name, reports_dir)
        auc_score = roc_auc_score(y_test, y_pred_prob)
        weighted_avg["roc_auc"] = auc_score
        print(f"ROC-AUC: {auc_score:.4f}")
    else:
        print(f"[WARNING] {model_name} does not support predict_proba → Skipping ROC")

    return weighted_avg
