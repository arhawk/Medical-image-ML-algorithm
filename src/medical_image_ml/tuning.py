from __future__ import annotations

from pathlib import Path

import keras_tuner as kt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from medical_image_ml.models import build_cnn
from medical_image_ml.paths import ensure_dir
from medical_image_ml.train import TrainingConfig, default_callbacks


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


def tune_random_forest(
    X_train,
    y_train,
    random_state: int = 42,
    quick: bool = False,
) -> tuple[RandomForestClassifier, dict]:
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

    rf = RandomForestClassifier(
        random_state=random_state,
        n_jobs=1 if quick else -1,
        class_weight="balanced",
    )
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=1 if quick else -1,
        verbose=1 if not quick else 0,
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, {
        "best_params": grid_search.best_params_,
        "best_cv_score": float(grid_search.best_score_),
    }


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
        model.add(layers.Dense(num_classes, activation="softmax"))
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
    tuner.search(
        X_train,
        y_train,
        epochs=min(config.epochs, 10),
        validation_data=(X_valid, y_valid),
        batch_size=config.batch_size,
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
