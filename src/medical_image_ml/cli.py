from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from medical_image_ml.error_analysis import run_error_analysis
from medical_image_ml.evaluate import (
    evaluate_classification,
    make_classification_report,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_sample_images,
    plot_training_history,
)
from medical_image_ml.gradcam import plot_gradcam_examples
from medical_image_ml.models import build_cnn, build_mlp, build_random_forest
from medical_image_ml.paths import ensure_dir, project_path, resolve_data_dir, resolve_output_dir, save_text
from medical_image_ml.preprocessing import prepare_dataset
from medical_image_ml.train import (
    configure_tensorflow_runtime,
    resolve_batch_size,
    TrainingConfig,
    fit_keras_model,
    fit_keras_model_final,
    fit_random_forest,
    predict_keras_model,
    set_seeds,
)
from medical_image_ml.tuning import (
    BEST_CNN_PARAMS,
    BEST_MLP_PARAMS,
    BEST_RF_PARAMS,
    rf_cache_path,
    tune_cnn,
    tune_random_forest,
)


def _figures_dir(output_dir: Path) -> Path:
    return ensure_dir(output_dir / "figures")


def _count_params(model) -> int:
    return int(model.count_params())


def _save_metrics(output_dir: Path, model_name: str, metrics: dict) -> None:
    save_text(output_dir / "metrics" / f"{model_name}.json", json.dumps(metrics, indent=2))


def _config_for_model(
    base: TrainingConfig,
    model_name: str,
    quick: bool,
    default_epochs: int,
) -> TrainingConfig:
    if quick:
        epochs = 5
    elif model_name == "cnn":
        epochs = 40
    elif model_name == "mlp":
        epochs = 30
    else:
        epochs = default_epochs
    return TrainingConfig(
        batch_size=base.batch_size,
        epochs=epochs,
        random_state=base.random_state,
        patience=base.patience,
        reduce_lr_patience=base.reduce_lr_patience,
    )


def train_rf(
    dataset,
    output_dir: Path,
    config: TrainingConfig,
    quick: bool = False,
    tune: bool = True,
    retune_rf: bool = False,
) -> tuple[dict, np.ndarray]:
    start = time.time()
    cache_file = rf_cache_path(output_dir)

    if tune:
        model, tuning_info = tune_random_forest(
            dataset.X_train_pca,
            dataset.y_train,
            random_state=config.random_state,
            quick=quick,
            cache_path=cache_file,
            reuse_cache=True,
            retune=retune_rf,
        )
        params = tuning_info["best_params"]
        metrics_extra = {
            "best_cv_score": tuning_info.get("best_cv_score"),
            "from_cache": tuning_info.get("from_cache", False),
        }
    else:
        model = build_random_forest(**BEST_RF_PARAMS, random_state=config.random_state)
        fit_random_forest(model, dataset.X_train_pca, dataset.y_train)
        params = BEST_RF_PARAMS
        metrics_extra = {"from_cache": False}

    X_final, y_final = dataset.merged_pca_arrays()
    fit_random_forest(model, X_final, y_final)

    y_pred = model.predict(dataset.X_test_pca)
    metrics = evaluate_classification(dataset.y_test, y_pred, title="Random Forest")
    metrics["params"] = params
    metrics.update(metrics_extra)
    metrics["train_seconds"] = round(time.time() - start, 2)
    metrics["param_count"] = sum(
        tree.tree_.node_count for tree in model.estimators_
    )

    plot_confusion_matrix(
        dataset.y_test,
        y_pred,
        title="Random Forest - Test Confusion Matrix",
        output_path=_figures_dir(output_dir) / "confusion_matrix_rf.png",
    )
    report = make_classification_report(dataset.y_test, y_pred)
    save_text(output_dir / "reports" / "rf_classification_report.txt", report)
    _save_metrics(output_dir, "rf", metrics)
    return metrics, y_pred


def train_mlp(
    dataset,
    output_dir: Path,
    config: TrainingConfig,
    quick: bool = False,
) -> tuple[dict, np.ndarray]:
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

    final_model = build_mlp(
        input_dim=dataset.X_train_pca.shape[1],
        num_classes=dataset.num_classes,
        **params,
    )
    X_final, y_final = dataset.merged_pca_arrays()
    fit_keras_model_final(final_model, X_final, y_final, config)

    y_pred = np.argmax(
        predict_keras_model(final_model, dataset.X_test_pca, batch_size=config.batch_size),
        axis=1,
    )
    metrics = evaluate_classification(dataset.y_test, y_pred, title="MLP")
    metrics["params"] = params
    metrics["train_seconds"] = round(time.time() - start, 2)
    metrics["param_count"] = _count_params(final_model)

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
    return metrics, y_pred


def train_cnn(
    dataset,
    output_dir: Path,
    config: TrainingConfig,
    quick: bool = False,
    tune: bool = False,
    run_gradcam: bool = True,
) -> tuple[dict, np.ndarray]:
    start = time.time()
    input_shape = dataset.X_train.shape[1:]

    if tune:
        _, tuning_info = tune_cnn(
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
        model = build_cnn(
            input_shape=input_shape,
            num_classes=dataset.num_classes,
            filters=(params.get("filter_1"), params.get("filter_2")),
            kernels=(params.get("kernel_1"), params.get("kernel_2")),
            dense_units=32,
            learning_rate=params.get("learning_rate", 1e-3),
        )
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

    if tune:
        final_params = params
        final_model = build_cnn(
            input_shape=input_shape,
            num_classes=dataset.num_classes,
            filters=(final_params.get("filter_1"), final_params.get("filter_2")),
            kernels=(final_params.get("kernel_1"), final_params.get("kernel_2")),
            dense_units=32,
            learning_rate=final_params.get("learning_rate", 1e-3),
        )
    else:
        final_model = build_cnn(
            input_shape=input_shape,
            num_classes=dataset.num_classes,
            **params,
        )

    X_final, y_final = dataset.merged_image_arrays()
    fit_keras_model_final(final_model, X_final, y_final, config)

    y_pred = np.argmax(
        predict_keras_model(final_model, dataset.X_test, batch_size=config.batch_size),
        axis=1,
    )
    metrics = evaluate_classification(dataset.y_test, y_pred, title="CNN")
    metrics["params"] = params
    metrics["train_seconds"] = round(time.time() - start, 2)
    metrics["param_count"] = _count_params(final_model)

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
    final_model.save(model_path)
    metrics["model_path"] = str(model_path)

    if run_gradcam:
        try:
            plot_gradcam_examples(
                final_model,
                dataset.X_test,
                dataset.y_test,
                y_pred,
                n_examples=4,
                output_path=_figures_dir(output_dir) / "gradcam_cnn.png",
                title="CNN Grad-CAM",
            )
        except Exception as exc:
            print(f"Warning: CNN Grad-CAM skipped ({exc})")

    return metrics, y_pred


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
        "gradcam_cnn.png",
        "error_misclassified_cnn.png",
        "error_analysis_rf_cnn.png",
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


def _save_predictions(output_dir: Path, model_name: str, y_pred: np.ndarray) -> None:
    pred_dir = ensure_dir(output_dir / "predictions")
    np.save(pred_dir / f"{model_name}_y_pred.npy", y_pred)


def _load_cached_predictions(output_dir: Path) -> dict[str, np.ndarray]:
    pred_dir = output_dir / "predictions"
    predictions: dict[str, np.ndarray] = {}
    if not pred_dir.exists():
        return predictions
    for path in pred_dir.glob("*_y_pred.npy"):
        name = path.name.replace("_y_pred.npy", "")
        predictions[name] = np.load(path)
    return predictions


def run_training(args: argparse.Namespace) -> list[dict]:
    runtime_info = configure_tensorflow_runtime(cpu_only=args.cpu_only)
    set_seeds(args.seed)
    data_dir = resolve_data_dir(args.data_dir)
    output_dir = ensure_dir(resolve_output_dir(args.output_dir))

    batch_size = resolve_batch_size(args.batch_size, gpu_count=int(runtime_info["gpu_count"]))
    base_config = TrainingConfig(
        batch_size=batch_size,
        epochs=args.epochs,
        random_state=args.seed,
    )

    dataset = prepare_dataset(data_dir=data_dir, random_state=args.seed)
    generate_dataset_figures(dataset, output_dir)
    print(
        "Runtime:",
        f"{runtime_info['cpu_threads']} CPU threads,",
        f"{runtime_info['gpu_count']} GPU(s),",
        f"batch_size={batch_size},",
        f"mixed_precision={runtime_info['mixed_precision']}",
    )
    if runtime_info["gpu_count"] > 0 and runtime_info.get("gpu_name"):
        print(f"  GPU device: {runtime_info['gpu_name']}")
    if runtime_info["gpu_count"] == 0:
        print(
            "Tip: install Apple GPU acceleration with "
            "`pip install tensorflow-metal` (or `pip install -e \".[macos]\"`)."
        )
    else:
        print("  Keras models (MLP/CNN) train on GPU via tensorflow-metal. RF uses CPU (scikit-learn).")

    def run_rf():
        cfg = _config_for_model(base_config, "rf", args.quick, args.epochs)
        return train_rf(
            dataset,
            output_dir,
            cfg,
            quick=args.quick,
            tune=not args.no_tune,
            retune_rf=args.retune_rf,
        )

    def run_mlp():
        cfg = _config_for_model(base_config, "mlp", args.quick, args.epochs)
        return train_mlp(dataset, output_dir, cfg, quick=args.quick)

    def run_cnn():
        cfg = _config_for_model(base_config, "cnn", args.quick, args.epochs)
        return train_cnn(
            dataset,
            output_dir,
            cfg,
            quick=args.quick,
            tune=args.tune,
        )

    runners = {
        "rf": run_rf,
        "mlp": run_mlp,
        "cnn": run_cnn,
    }

    if args.model == "all":
        models = ["rf", "mlp", "cnn"]
    elif args.model == "portfolio":
        models = ["rf", "cnn"]
    else:
        models = [args.model]

    results: list[dict] = []
    predictions: dict[str, np.ndarray] = {}

    for name in models:
        print(f"\n=== Training {name.upper()} ===")
        metrics, y_pred = runners[name]()
        predictions[name] = y_pred
        _save_predictions(output_dir, name, y_pred)
        row = {
            "model": name.upper(),
            "accuracy": metrics["accuracy"],
            "f1_weighted": metrics["f1_weighted"],
            "train_seconds": metrics["train_seconds"],
        }
        if "param_count" in metrics:
            row["param_count"] = metrics["param_count"]
        results.append(row)
        print(
            f"{name.upper()} test accuracy: {metrics['accuracy']:.4f} "
            f"(trained in {metrics['train_seconds']}s, "
            f"params={metrics.get('param_count', 'n/a')})"
        )

    if len(results) > 1:
        comparison_path = _figures_dir(output_dir) / "model_comparison.png"
        plot_model_comparison(results, output_path=comparison_path)
        save_text(output_dir / "metrics" / "comparison.json", json.dumps(results, indent=2))

    if "cnn" in predictions or "rf" in predictions:
        cached_preds = _load_cached_predictions(output_dir)
        merged_preds = {**cached_preds, **predictions}
        run_error_analysis(dataset.X_test, dataset.y_test, merged_preds, output_dir)

    _sync_showcase_assets(output_dir)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train medical image classification models (RF, MLP, CNN)."
    )
    parser.add_argument(
        "--model",
        choices=["rf", "mlp", "cnn", "all", "portfolio"],
        default="all",
        help=(
            "Model to train. 'all' = Phase 1 baseline (rf, mlp, cnn). "
            "'portfolio' = rf + cnn with Grad-CAM and error analysis."
        ),
    )
    parser.add_argument("--data-dir", default=None, help="Path to Assignment2Data directory.")
    parser.add_argument("--output-dir", default=None, help="Directory for artifacts.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30, help="Default epochs (CNN uses 40, MLP 30 when not --quick).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Mini-batch size for Keras (0 = auto: 256 with GPU, 128 on 8+ CPU cores, else 64).",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Disable GPU and run TensorFlow on CPU only.",
    )
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
    parser.add_argument(
        "--retune-rf",
        action="store_true",
        help="Force RF GridSearch even when outputs/tuning/rf/best_params.json exists.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
