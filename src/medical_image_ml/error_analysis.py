from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from medical_image_ml.paths import ensure_dir, save_text


def top_confusion_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_k: int = 5,
) -> list[dict]:
    """Return the most frequent off-diagonal confusion pairs."""
    cm = confusion_matrix(y_true, y_pred)
    pairs: list[dict] = []
    for true_cls in range(cm.shape[0]):
        for pred_cls in range(cm.shape[1]):
            if true_cls != pred_cls and cm[true_cls, pred_cls] > 0:
                pairs.append(
                    {
                        "true_class": int(true_cls),
                        "pred_class": int(pred_cls),
                        "count": int(cm[true_cls, pred_cls]),
                    }
                )
    pairs.sort(key=lambda p: p["count"], reverse=True)
    return pairs[:top_k]


def compare_model_failures(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    name_a: str = "RF",
    name_b: str = "CNN",
) -> dict:
    """Summarize where two models disagree on test errors."""
    wrong_a = y_pred_a != y_true
    wrong_b = y_pred_b != y_true
    return {
        f"{name_a}_only_wrong": int(np.sum(wrong_a & ~wrong_b)),
        f"{name_b}_only_wrong": int(np.sum(wrong_b & ~wrong_a)),
        "both_wrong": int(np.sum(wrong_a & wrong_b)),
        "both_correct": int(np.sum(~wrong_a & ~wrong_b)),
        f"{name_a}_accuracy": float(np.mean(~wrong_a)),
        f"{name_b}_accuracy": float(np.mean(~wrong_b)),
    }


def plot_misclassified_examples(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n: int = 3,
    output_path: str | Path | None = None,
    title: str = "Misclassified Examples",
) -> Path | None:
    """Plot n misclassified test images with true vs predicted labels."""
    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        return None

    chosen = wrong_idx[:n]
    fig, axes = plt.subplots(1, len(chosen), figsize=(4 * len(chosen), 4))
    if len(chosen) == 1:
        axes = [axes]

    for ax, idx in zip(axes, chosen):
        ax.imshow(X[idx])
        ax.set_title(f"true={y_true[idx]}, pred={y_pred[idx]}")
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    if output_path is None:
        plt.close(fig)
        return None

    target = ensure_dir(Path(output_path).parent) / Path(output_path).name
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_failure_comparison(
    failure_stats: dict,
    output_path: str | Path | None = None,
) -> Path | None:
    """Bar chart of RF-only vs CNN-only vs both-wrong test samples."""
    keys = [k for k in failure_stats if k.endswith("_only_wrong") or k == "both_wrong"]
    labels = [k.replace("_only_wrong", " only").replace("_", " ").title() for k in keys]
    values = [failure_stats[k] for k in keys]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=labels, y=values, hue=labels, ax=ax, palette="Set2", legend=False)
    ax.set_ylabel("Test samples")
    ax.set_title("RF vs CNN Failure Overlap")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")
    plt.tight_layout()

    if output_path is None:
        plt.close(fig)
        return None

    target = ensure_dir(Path(output_path).parent) / Path(output_path).name
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target


def format_error_analysis_report(
    cnn_pairs: list[dict],
    failure_stats: dict | None = None,
    transfer_pairs: list[dict] | None = None,
) -> str:
    """Build a text report summarizing error analysis findings."""
    lines = ["Error Analysis Report", "=" * 40, ""]

    lines.append("CNN — top confused class pairs (true → predicted):")
    for pair in cnn_pairs:
        lines.append(
            f"  Class {pair['true_class']} → {pair['pred_class']}: "
            f"{pair['count']} samples"
        )
    lines.append("")

    if transfer_pairs:
        lines.append("Transfer — top confused class pairs:")
        for pair in transfer_pairs:
            lines.append(
                f"  Class {pair['true_class']} → {pair['pred_class']}: "
                f"{pair['count']} samples"
            )
        lines.append("")

    if failure_stats:
        lines.append("RF vs CNN failure comparison:")
        for key, value in failure_stats.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")

    return "\n".join(lines) + "\n"


def run_error_analysis(
    X_test: np.ndarray,
    y_test: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_dir: Path,
) -> dict:
    """Generate error-analysis figures and report from model predictions."""
    fig_dir = ensure_dir(output_dir / "figures")
    report_dir = ensure_dir(output_dir / "reports")
    findings: dict = {}

    if "cnn" in predictions:
        y_pred_cnn = predictions["cnn"]
        cnn_pairs = top_confusion_pairs(y_test, y_pred_cnn)
        findings["cnn_confusion_pairs"] = cnn_pairs

        plot_misclassified_examples(
            X_test,
            y_test,
            y_pred_cnn,
            n=3,
            output_path=fig_dir / "error_misclassified_cnn.png",
            title="CNN Misclassified Examples",
        )

    if "transfer" in predictions:
        y_pred_transfer = predictions["transfer"]
        findings["transfer_confusion_pairs"] = top_confusion_pairs(y_test, y_pred_transfer)

    failure_stats = None
    if "rf" in predictions and "cnn" in predictions:
        failure_stats = compare_model_failures(
            y_test, predictions["rf"], predictions["cnn"]
        )
        findings["rf_vs_cnn"] = failure_stats
        plot_failure_comparison(failure_stats, output_path=fig_dir / "error_analysis_rf_cnn.png")

    report = format_error_analysis_report(
        cnn_pairs=findings.get("cnn_confusion_pairs", []),
        failure_stats=failure_stats,
        transfer_pairs=findings.get("transfer_confusion_pairs"),
    )
    save_text(report_dir / "error_analysis.txt", report)
    return findings
