from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import keras_tuner as kt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from medical_image_ml.models import build_cnn, build_random_forest
from medical_image_ml.paths import ensure_dir, save_text
from medical_image_ml.train import TrainingConfig, default_callbacks, make_tf_dataset


BEST_RF_PARAMS = {
    "n_estimators": 250,
    "max_depth": 20,
    "max_features": "sqrt",
}

BEST_MLP_PARAMS = {
    "units": 256,
    "activation": "relu",
    "dropout": 0.2,
    "learning_rate": 0.1,
    "hidden_layers": 2,
}

BEST_CNN_PARAMS = {
    "filters": (32, 64),
    "kernels": (3, 5),
    "dense_units": 64,
    "dropout_conv": 0.3,
    "dropout_dense": 0.3,
    "learning_rate": 1e-3,
}

RF_CACHE_FILENAME = "best_params.json"


def rf_cache_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "tuning" / "rf" / RF_CACHE_FILENAME


def _normalize_rf_params(params: dict) -> dict:
    normalized = dict(params)
    if normalized.get("max_depth") is None:
        normalized["max_depth"] = None
    return normalized


def load_rf_cache(cache_path: Path) -> dict | None:
    if not cache_path.exists():
        return None
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    data["best_params"] = _normalize_rf_params(data["best_params"])
    return data


def save_rf_cache(cache_path: Path, best_params: dict, best_cv_score: float) -> None:
    payload = {
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_text(cache_path, json.dumps(payload, indent=2))


def build_rf_from_params(params: dict, random_state: int = 42) -> RandomForestClassifier:
    rf_params = _normalize_rf_params(params)
    return build_random_forest(
        n_estimators=rf_params["n_estimators"],
        max_depth=rf_params.get("max_depth"),
        max_features=rf_params["max_features"],
        random_state=random_state,
    )


def tune_random_forest(
    X_train,
    y_train,
    random_state: int = 42,
    quick: bool = False,
    cache_path: str | Path | None = None,
    reuse_cache: bool = True,
    retune: bool = False,
) -> tuple[RandomForestClassifier, dict]:
    if cache_path is not None and reuse_cache and not retune:
        cached = load_rf_cache(Path(cache_path))
        if cached is not None:
            print(f"Loaded cached RF params from {cache_path}")
            model = build_rf_from_params(cached["best_params"], random_state=random_state)
            return model, {
                "best_params": cached["best_params"],
                "best_cv_score": cached.get("best_cv_score"),
                "from_cache": True,
            }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    if quick:
        param_grid = {
            "n_estimators": [100, 150],
            "max_depth": [20],
            "max_features": ["sqrt"],
        }
    else:
        param_grid = {
            "n_estimators": [100, 150, 200, 250],
            "max_depth": [None, 20, 40],
            "max_features": ["sqrt", "log2"],
        }

    grid_parallel = 1 if quick else -1
    # Avoid nested n_jobs=-1 (GridSearch workers × RF trees) which thrashes CPU caches.
    rf_parallel = 1 if grid_parallel != 1 else -1

    rf = RandomForestClassifier(
        random_state=random_state,
        n_jobs=rf_parallel,
        class_weight="balanced",
    )
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=grid_parallel,
        verbose=1 if not quick else 0,
    )
    grid_search.fit(X_train, y_train)
    best_params = _normalize_rf_params(grid_search.best_params_)
    tuning_info = {
        "best_params": best_params,
        "best_cv_score": float(grid_search.best_score_),
        "from_cache": False,
    }

    if cache_path is not None:
        cache_file = Path(cache_path)
        ensure_dir(cache_file.parent)
        save_rf_cache(cache_file, best_params, tuning_info["best_cv_score"])
        print(f"Saved RF tuning cache to {cache_file}")

    return grid_search.best_estimator_, tuning_info


def tune_cnn(
    X_train,
    y_train,
    X_valid,
    y_valid,
    input_shape: tuple[int, int, int],
    num_classes: int,
    output_dir: str | Path,
    config: TrainingConfig,
    max_trials: int = 5,
) -> tuple[keras.Model, dict]:
    l2_reg = regularizers.l2(1e-4)
    filter_nums = [16, 32, 64]
    kernel_sizes = [3, 5]

    def build_model(hp):
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=input_shape))
        for i in range(1, 3):
            model.add(
                layers.Conv2D(
                    filters=hp.Choice(f"filter_{i}", values=filter_nums),
                    kernel_size=hp.Choice(f"kernel_{i}", values=kernel_sizes),
                    padding="same",
                    activation="relu",
                    kernel_regularizer=l2_reg,
                )
            )
            model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(rate=0.4))
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(32, activation="relu", kernel_regularizer=l2_reg))
        model.add(layers.Dropout(rate=0.4))
        model.add(layers.Dense(num_classes, activation="softmax", dtype="float32"))
        learning_rate = hp.Choice("learning_rate", values=[1e-3, 1e-2])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    tuning_dir = ensure_dir(Path(output_dir) / "tuning" / "cnn")
    tuner = kt.RandomSearch(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=max_trials,
        executions_per_trial=1,
        directory=str(tuning_dir),
        project_name="cnn_search",
        overwrite=True,
    )
    train_ds = make_tf_dataset(X_train, y_train, batch_size=config.batch_size, training=True)
    valid_ds = make_tf_dataset(X_valid, y_valid, batch_size=config.batch_size, training=False)
    tuner.search(
        train_ds,
        epochs=min(config.epochs, 10),
        validation_data=valid_ds,
        callbacks=default_callbacks(config.patience, config.reduce_lr_patience),
        verbose=1,
    )
    best_hp = tuner.get_best_hyperparameters(1)[0]
    model = build_cnn(
        input_shape=input_shape,
        num_classes=num_classes,
        filters=(best_hp.get("filter_1"), best_hp.get("filter_2")),
        kernels=(best_hp.get("kernel_1"), best_hp.get("kernel_2")),
        dense_units=32,
        learning_rate=best_hp.get("learning_rate"),
    )
    return model, {"best_hyperparameters": best_hp.values}
