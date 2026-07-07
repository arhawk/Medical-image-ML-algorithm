from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from medical_image_ml.evaluate import (
    evaluate_classification,
    make_classification_report,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_sample_images,
    plot_training_history,
)
from medical_image_ml.models import build_cnn, build_mlp, build_random_forest
from medical_image_ml.paths import ensure_dir, project_path, resolve_data_dir, resolve_output_dir, save_text
from medical_image_ml.preprocessing import prepare_dataset
from medical_image_ml.train import TrainingConfig, fit_keras_model, fit_random_forest, set_seeds
from medical_image_ml.tuning import (
    BEST_CNN_PARAMS,
    BEST_MLP_PARAMS,
    BEST_RF_PARAMS,
    tune_cnn,
    tune_random_forest,
)


def _figures_dir(output_dir: Path) -> Path:
    return ensure_dir(output_dir / "figures")


def _save_metrics(output_dir: Path, model_name: str, metrics: dict) -> None:
    save_text(output_dir / "metrics" / f"{model_name}.json", json.dumps(metrics, indent=2))


def train_rf(
    dataset,
    output_dir: Path,
    config: TrainingConfig,
    quick: bool = False,
    tune: bool = True,
) -> dict:
    start = time.time()
    if tune:
        model, tuning_info = tune_random_forest(
            dataset.X_train_pca,
            dataset.y_train,
            random_state=config.random_state,
            quick=quick,
        )
        params = tuning_info["best_params"]
    else:
        model = build_random_forest(**BEST_RF_PARAMS, random_state=config.random_state)
        fit_random_forest(model, dataset.X_train_pca, dataset.y_train)
        params = BEST_RF_PARAMS

    X_final = np.concatenate([dataset.X_train_pca, dataset.X_valid_pca])
    y_final = np.concatenate([dataset.y_train, dataset.y_valid])
    fit_random_forest(model, X_final, y_final)

    y_pred = model.predict(dataset.X_test_pca)
    metrics = evaluate_classification(dataset.y_test, y_pred, title="Random Forest")
    metrics["params"] = params
    metrics["train_seconds"] = round(time.time() - start, 2)

    plot_confusion_matrix(
        dataset.y_test,
        y_pred,
        title="Random Forest - Test Confusion Matrix",
        output_path=_figures_dir(output_dir) / "confusion_matrix_rf.png",
    )
    report = make_classification_report(dataset.y_test, y_pred)
    save_text(output_dir / "reports" / "rf_classification_report.txt", report)
    _save_metrics(output_dir, "rf", metrics)
    return metrics


def train_mlp(
    dataset,
    output_dir: Path,
    config: TrainingConfig,
    quick: bool = False,
) -> dict:
    start = time.time()
    params = BEST_MLP_PARAMS.copy()
    if quick:
        params["units"] = 128
        params["hidden_layers"] = 1

    model = build_mlp(
        input_dim=dataset.X_train_pca.shape[1],
        num_classes=dataset.num_classes,
        **params,
    )
    history = fit_keras_model(
        model,
        dataset.X_train_pca,
        dataset.y_train,
        dataset.X_valid_pca,
        dataset.y_valid,
        config,
    )

    y_pred = np.argmax(model.predict(dataset.X_test_pca, verbose=0), axis=1)
    metrics = evaluate_classification(dataset.y_test, y_pred, title="MLP")
    metrics["params"] = params
    metrics["train_seconds"] = round(time.time() - start, 2)

    plot_training_history(
        history,
        title="MLP Training History",
        output_path=_figures_dir(output_dir) / "history_mlp.png",
    )
    plot_confusion_matrix(
        dataset.y_test,
        y_pred,
        title="MLP - Test Confusion Matrix",
        output_path=_figures_dir(output_dir) / "confusion_matrix_mlp.png",
    )
    report = make_classification_report(dataset.y_test, y_pred)
    save_text(output_dir / "reports" / "mlp_classification_report.txt", report)
    _save_metrics(output_dir, "mlp", metrics)
    return metrics


def train_cnn(
    dataset,
    output_dir: Path,
    config: TrainingConfig,
    quick: bool = False,
    tune: bool = False,
) -> dict:
    start = time.time()
    input_shape = dataset.X_train.shape[1:]

    if tune:
        model, tuning_info = tune_cnn(
            dataset.X_train,
            dataset.y_train,
            dataset.X_valid,
            dataset.y_valid,
            input_shape=input_shape,
            num_classes=dataset.num_classes,
            output_dir=output_dir,
            config=config,
            max_trials=3 if quick else 8,
        )
        params = tuning_info["best_hyperparameters"]
    else:
        params = BEST_CNN_PARAMS.copy()
        if quick:
            params["filters"] = (16, 32)
            params["dense_units"] = 32
        model = build_cnn(
            input_shape=input_shape,
            num_classes=dataset.num_classes,
            **params,
        )

    history = fit_keras_model(
        model,
        dataset.X_train,
        dataset.y_train,
        dataset.X_valid,
        dataset.y_valid,
        config,
    )

    y_pred = np.argmax(model.predict(dataset.X_test, verbose=0), axis=1)
    metrics = evaluate_classification(dataset.y_test, y_pred, title="CNN")
    metrics["params"] = params
    metrics["train_seconds"] = round(time.time() - start, 2)

    plot_training_history(
        history,
        title="CNN Training History",
        output_path=_figures_dir(output_dir) / "history_cnn.png",
    )
    plot_confusion_matrix(
        dataset.y_test,
        y_pred,
        title="CNN - Test Confusion Matrix",
        output_path=_figures_dir(output_dir) / "confusion_matrix_cnn.png",
    )
    report = make_classification_report(dataset.y_test, y_pred)
    save_text(output_dir / "reports" / "cnn_classification_report.txt", report)
    _save_metrics(output_dir, "cnn", metrics)

    model_path = ensure_dir(output_dir / "models") / "cnn.keras"
    model.save(model_path)
    metrics["model_path"] = str(model_path)
    return metrics


def _sync_showcase_assets(output_dir: Path) -> None:
    docs_assets = ensure_dir(project_path("docs", "assets"))
    fig_dir = output_dir / "figures"
    if not fig_dir.exists():
        return
    for name in [
        "sample_images.png",
        "class_distribution.png",
        "model_comparison.png",
        "confusion_matrix_rf.png",
        "confusion_matrix_mlp.png",
        "confusion_matrix_cnn.png",
        "history_mlp.png",
        "history_cnn.png",
    ]:
        src = fig_dir / name
        if src.exists():
            (docs_assets / name).write_bytes(src.read_bytes())


def generate_dataset_figures(dataset, output_dir: Path) -> None:
    fig_dir = _figures_dir(output_dir)
    docs_assets = ensure_dir(project_path("docs", "assets"))
    plot_sample_images(
        dataset.X_train,
        dataset.y_train,
        output_path=fig_dir / "sample_images.png",
    )
    plot_class_distribution(
        dataset.y_train,
        output_path=fig_dir / "class_distribution.png",
    )
    for name in ["sample_images.png", "class_distribution.png"]:
        src = fig_dir / name
        if src.exists():
            (docs_assets / name).write_bytes(src.read_bytes())


def run_training(args: argparse.Namespace) -> list[dict]:
    set_seeds(args.seed)
    data_dir = resolve_data_dir(args.data_dir)
    output_dir = ensure_dir(resolve_output_dir(args.output_dir))

    config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=5 if args.quick else args.epochs,
        random_state=args.seed,
    )

    dataset = prepare_dataset(data_dir=data_dir, random_state=args.seed)
    generate_dataset_figures(dataset, output_dir)

    runners = {
        "rf": lambda: train_rf(dataset, output_dir, config, quick=args.quick, tune=not args.no_tune),
        "mlp": lambda: train_mlp(dataset, output_dir, config, quick=args.quick),
        "cnn": lambda: train_cnn(
            dataset,
            output_dir,
            config,
            quick=args.quick,
            tune=args.tune,
        ),
    }

    models = ["rf", "mlp", "cnn"] if args.model == "all" else [args.model]
    results: list[dict] = []

    for name in models:
        print(f"\n=== Training {name.upper()} ===")
        metrics = runners[name]()
        row = {
            "model": name.upper(),
            "accuracy": metrics["accuracy"],
            "f1_weighted": metrics["f1_weighted"],
            "train_seconds": metrics["train_seconds"],
        }
        results.append(row)
        print(
            f"{name.upper()} test accuracy: {metrics['accuracy']:.4f} "
            f"(trained in {metrics['train_seconds']}s)"
        )

    if len(results) > 1:
        comparison_path = _figures_dir(output_dir) / "model_comparison.png"
        plot_model_comparison(results, output_path=comparison_path)
        docs_assets = ensure_dir(project_path("docs", "assets"))
        if comparison_path.exists():
            (docs_assets / "model_comparison.png").write_bytes(comparison_path.read_bytes())
        for item in results:
            item["model"] = item["model"]
        save_text(output_dir / "metrics" / "comparison.json", json.dumps(results, indent=2))

    _sync_showcase_assets(output_dir)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train medical image classification models (RF, MLP, CNN)."
    )
    parser.add_argument(
        "--model",
        choices=["rf", "mlp", "cnn", "all"],
        default="all",
        help="Model to train.",
    )
    parser.add_argument("--data-dir", default=None, help="Path to Assignment2Data directory.")
    parser.add_argument("--output-dir", default=None, help="Directory for artifacts.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke-test run with reduced epochs and smaller search spaces.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable keras-tuner search for CNN (ignored for RF/MLP unless --model cnn).",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip GridSearch for RF and use best-known hyperparameters.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
