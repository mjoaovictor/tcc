import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from matplotlib import pyplot as plt
import seaborn as sns

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple:
    thresholds = np.linspace(0.01, 0.99, 200)

    best_threshold: float = 0.5
    best_f1: float = -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1


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
    print(classification_report(y_true, y_pred, zero_division=0))

    return metrics


def evaluate_model_v2(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float] | None = None
) -> tuple:
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    pr_auc = average_precision_score(y_true, y_prob)

    print(f"{'='*65}")
    print(f"Análise de Desempenho Clínico (PR-AUC: {pr_auc:.4f})")
    print(f"{'='*65}\n")

    # 2. Análise de múltiplos Thresholds (Precision, Recall, F1)
    threshold_data = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)

        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        # Métrica de negócio/clínica: Quantos alarmes falsos para cada caso real?
        false_positives = ((preds == 1) & (y_true == 0)).sum()
        true_positives = ((preds == 1) & (y_true == 1)).sum()
        ratio_fp_tp = false_positives / true_positives if true_positives > 0 else np.nan

        threshold_data.append({
            "Threshold": t,
            "Precision": f"{prec:.4%}",
            "Recall": f"{rec:.4%}",
            "F1-Score": f"{f1:.4f}",
            "Falsos Positivos / Positivo Real": f"{ratio_fp_tp:.1f}"
        })

    df_thresholds = pd.DataFrame(threshold_data)
    print("Métricas por Limiar de Decisão:")
    print(df_thresholds.to_string(index=False))
    print("\n" + "-"*65 + "\n")

    # 3. Geração dos Gráficos de Diagnóstico
    sns.set_theme(style="ticks", palette="colorblind")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = sns.color_palette("colorblind")

    # Gráfico A: Curva Precision-Recall Completa
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    baseline = np.mean(y_true)

    axes[0].plot(recalls, precisions, color=colors[0], lw=2, label=f"Modelo (PR-AUC = {pr_auc:.3f})")
    axes[0].axhline(y=baseline, color=colors[1], linestyle="--", lw=1.5, label=f"Baseline ({baseline:.2%})")

    axes[0].set_title("Curva Precision-Recall")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precisão")
    axes[0].legend(loc="upper right")
    axes[0].set_xlim([-0.02, 1.02])
    axes[0].set_ylim([-0.02, 1.02])

    # axes[0].grid(True, linestyle=":", alpha=0.6)

    # Gráfico B: Distribuição das Probabilidades Preditas (Diagnóstico de Superestimação)
    labels_text = np.where(y_true == 1, "Sim (AVC)", "Não (Saudável)")
    df_dist = pd.DataFrame({"Probabilidade": y_prob, "Classe": labels_text})
    sns.histplot(
        data=df_dist,
        x="Probabilidade",
        hue="Classe",
        element="step",
        stat="density",
        common_norm=False,
        bins=50,
        kde=True,
        ax=axes[1],
        # palette=["#4C72B0", "#C44E52"],
        alpha=0.3,
        hue_order=["Não (Saudável)", "Sim (AVC)"]
    )
    axes[1].set_title("Distribuição das Probabilidades Preditas")
    axes[1].set_xlabel("Probabilidade Estimada pelo Modelo")
    axes[1].set_ylabel("Densidade")
    axes[1].set_xlim(0, 1)
    # axes[1].grid(True, linestyle=":", alpha=0.6)

    for ax in axes:
        sns.despine(ax=ax)

    plt.tight_layout()
    plt.show()

    return pr_auc, df_thresholds
