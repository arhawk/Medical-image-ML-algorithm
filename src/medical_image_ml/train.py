from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from tensorflow.keras import mixed_precision
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


def configure_cpu_environment() -> int:
    """Set BLAS/OpenMP thread counts once so sklearn and TensorFlow do not oversubscribe."""
    cpu_count = os.cpu_count() or 1
    thread_vars = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    for name in thread_vars:
        os.environ.setdefault(name, str(cpu_count))
    return cpu_count


def configure_tensorflow_runtime(
    enable_mixed_precision: bool = True,
    cpu_only: bool = False,
) -> dict[str, int | bool | str]:
    cpu_count = configure_cpu_environment()

    try:
        tf.config.threading.set_intra_op_parallelism_threads(cpu_count)
        tf.config.threading.set_inter_op_parallelism_threads(max(2, cpu_count // 2))
    except RuntimeError:
        # TensorFlow locks threading settings after the first runtime use.
        pass

    if cpu_only:
        try:
            tf.config.set_visible_devices([], "GPU")
        except (RuntimeError, ValueError):
            pass

    gpu_devices = tf.config.list_physical_devices("GPU")
    for device in gpu_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except (RuntimeError, ValueError):
            pass

    mixed_precision_enabled = False
    if enable_mixed_precision and gpu_devices and not cpu_only:
        mixed_precision.set_global_policy("mixed_float16")
        mixed_precision_enabled = True
    else:
        mixed_precision.set_global_policy("float32")

    gpu_name = ""
    if gpu_devices:
        try:
            gpu_name = gpu_devices[0].name
        except (AttributeError, IndexError):
            gpu_name = "GPU"

    return {
        "cpu_threads": cpu_count,
        "gpu_count": len(gpu_devices),
        "mixed_precision": mixed_precision_enabled,
        "gpu_name": gpu_name,
    }


def resolve_batch_size(batch_size: int, gpu_count: int = 0) -> int:
    """Pick a mini-batch size tuned for this 28x28 dataset and hardware."""
    if batch_size > 0:
        return batch_size
    if gpu_count > 0:
        return 256
    cpu_count = os.cpu_count() or 1
    if cpu_count >= 8:
        return 128
    return 64


def make_tf_dataset(
    X,
    y=None,
    batch_size: int = 128,
    training: bool = False,
    cache: bool = True,
) -> tf.data.Dataset:
    if y is None:
        dataset = tf.data.Dataset.from_tensor_slices(X)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if cache:
        dataset = dataset.cache()

    if training:
        shuffle_size = min(len(X), max(batch_size * 8, 1024))
        dataset = dataset.shuffle(shuffle_size, reshuffle_each_iteration=True)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


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
    train_ds = make_tf_dataset(X_train, y_train, batch_size=config.batch_size, training=True)
    valid_ds = make_tf_dataset(X_valid, y_valid, batch_size=config.batch_size, training=False)
    return model.fit(
        train_ds,
        epochs=config.epochs,
        validation_data=valid_ds,
        callbacks=callbacks,
        verbose=1,
    )


def fit_keras_model_final(
    model: keras.Model,
    X,
    y,
    config: TrainingConfig,
) -> keras.callbacks.History:
    train_ds = make_tf_dataset(X, y, batch_size=config.batch_size, training=True)
    return model.fit(
        train_ds,
        epochs=config.epochs,
        verbose=1,
    )


def predict_keras_model(
    model: keras.Model,
    X,
    batch_size: int = 128,
) -> np.ndarray:
    dataset = make_tf_dataset(X, batch_size=batch_size, training=False)
    return model.predict(dataset, verbose=0)


def fit_random_forest(model: RandomForestClassifier, X_train, y_train) -> RandomForestClassifier:
    model.fit(X_train, y_train)
    return model
