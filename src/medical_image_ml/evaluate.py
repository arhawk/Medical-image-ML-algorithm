from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from medical_image_ml.paths import ensure_dir


def evaluate_classification(
    y_true,
    y_pred,
    title: str = "Model",
) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "title": title,
    }


def make_classification_report(y_true, y_pred, digits: int = 3) -> str:
    return classification_report(y_true, y_pred, digits=digits)


def plot_confusion_matrix(
    y_true,
    y_pred,
    title: str = "Confusion Matrix",
    output_path: str | Path | None = None,
    class_names: list[str] | None = None,
) -> Path | None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        cmap="Blues",
        values_format="d",
        ax=ax,
    )
    ax.set_title(title)
    plt.tight_layout()

    if output_path is None:
        plt.close(fig)
        return None

    target = ensure_dir(Path(output_path).parent) / Path(output_path).name
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_training_history(
    history,
    title: str = "Training History",
    output_path: str | Path | None = None,
) -> Path | None:
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(history.history["loss"], label="Train Loss")
    axs[0].plot(history.history["val_loss"], label="Val Loss")
    axs[0].set_title("Loss")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(history.history["accuracy"], label="Train Accuracy")
    axs[1].plot(history.history["val_accuracy"], label="Val Accuracy")
    axs[1].set_title("Accuracy")
    axs[1].legend()
    axs[1].grid(True)

    fig.suptitle(title)
    plt.tight_layout()

    if output_path is None:
        plt.close(fig)
        return None

    target = ensure_dir(Path(output_path).parent) / Path(output_path).name
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_model_comparison(
    results: list[dict],
    output_path: str | Path | None = None,
) -> Path | None:
    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="model", y="accuracy", hue="model", ax=ax, palette="viridis", legend=False)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test Accuracy")
    ax.set_xlabel("Model")
    ax.set_title("Model Comparison")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f")
    plt.tight_layout()

    if output_path is None:
        plt.close(fig)
        return None

    target = ensure_dir(Path(output_path).parent) / Path(output_path).name
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_sample_images(
    X: np.ndarray,
    y: np.ndarray,
    n_per_class: int = 5,
    output_path: str | Path | None = None,
) -> Path | None:
    classes = np.unique(y)
    n_cols = n_per_class
    n_rows = len(classes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, cls in enumerate(classes):
        idxs = np.where(y == cls)[0][:n_per_class]
        for col, idx in enumerate(idxs):
            axes[row, col].imshow(X[idx])
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(f"Class {cls}", rotation=0, labelpad=30)

    fig.suptitle("Sample Images per Class")
    plt.tight_layout()

    if output_path is None:
        plt.close(fig)
        return None

    target = ensure_dir(Path(output_path).parent) / Path(output_path).name
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_class_distribution(
    y: np.ndarray,
    output_path: str | Path | None = None,
) -> Path | None:
    fig, ax = plt.subplots(figsize=(8, 5))
    pd.Series(y).value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.tight_layout()

    if output_path is None:
        plt.close(fig)
        return None

    target = ensure_dir(Path(output_path).parent) / Path(output_path).name
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target
