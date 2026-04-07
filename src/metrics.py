import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
        y_true: pd.Series | np.ndarray,
        y_prob: pd.Series | np.ndarray,
        threshold: float = 0.5
    ) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "PR-AUC": average_precision_score(y_true, y_prob),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "Brier_Score": brier_score_loss(y_true, y_prob),  # calibration metric
        "F1_Score": float(f1_score(y_true, y_pred)),
        "Recall": float(recall_score(y_true, y_pred)),
        "Confusion_Matrix": cm,
    }

    print("-" * 40)
    print(f"PR-AUC: {metrics['PR-AUC']:.4f}")
    print(f"ROC-AUC: {metrics['ROC-AUC']:.4f}")
    print(f"Brier Score (Calibration): {metrics['Brier_Score']:.4f}")
    print("-" * 40)
    print(f"Metrics at Threshold = {threshold}:")
    print(f"F1-Score:                  {metrics['F1_Score']:.4f}")
    print(f"Recall:                    {metrics['Recall']:.4f}")
    print("-" * 40)
    print("Confusion Matrix:")
    print(f"TN: {cm[0, 0]:<6} | FP: {cm[0, 1]}")
    print(f"FN: {cm[1, 0]:<6} | TP: {cm[1, 1]}")
    print("-" * 40)
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    return metrics
