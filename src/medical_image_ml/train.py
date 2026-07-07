from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


@dataclass
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 30
    random_state: int = 42
    patience: int = 4
    reduce_lr_patience: int = 2


def set_seeds(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def default_callbacks(patience: int = 4, reduce_lr_patience: int = 2) -> list[keras.callbacks.Callback]:
    return [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=reduce_lr_patience,
            verbose=1,
        ),
    ]


def fit_keras_model(
    model: keras.Model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    config: TrainingConfig,
    extra_callbacks: list[keras.callbacks.Callback] | None = None,
) -> keras.callbacks.History:
    callbacks = default_callbacks(config.patience, config.reduce_lr_patience)
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    return model.fit(
        X_train,
        y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_data=(X_valid, y_valid),
        callbacks=callbacks,
        verbose=1,
    )


def fit_random_forest(model: RandomForestClassifier, X_train, y_train) -> RandomForestClassifier:
    model.fit(X_train, y_train)
    return model
