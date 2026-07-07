from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from medical_image_ml.paths import resolve_data_dir


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    X_valid: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray
    X_train_pca: np.ndarray
    X_valid_pca: np.ndarray
    X_test_pca: np.ndarray
    pca: PCA
    num_classes: int


def load_arrays(data_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = resolve_data_dir(data_dir)
    X_train = np.load(root / "X_train.npy")
    X_test = np.load(root / "X_test.npy")
    y_train = np.load(root / "y_train.npy")
    y_test = np.load(root / "y_test.npy")
    return X_train, X_test, y_train, y_test


def normalize_images(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return X_train / 255.0, X_test / 255.0


def split_train_validation(
    X: np.ndarray,
    y: np.ndarray,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        X,
        y,
        test_size=validation_size,
        shuffle=True,
        stratify=y,
        random_state=random_state,
    )


def flatten_images(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def fit_pca(X_train_flat: np.ndarray, n_components: float = 0.95) -> PCA:
    pca = PCA(n_components=n_components)
    pca.fit(X_train_flat)
    return pca


def transform_pca(pca: PCA, X_flat: np.ndarray) -> np.ndarray:
    return pca.transform(X_flat)


def prepare_dataset(
    data_dir: str | Path | None = None,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> DatasetBundle:
    X_train_full, X_test, y_train_full, y_test = load_arrays(data_dir)
    X_train_full, X_test = normalize_images(X_train_full, X_test)
    X_train, X_valid, y_train, y_valid = split_train_validation(
        X_train_full,
        y_train_full,
        validation_size=validation_size,
        random_state=random_state,
    )

    X_train_flat = flatten_images(X_train)
    X_valid_flat = flatten_images(X_valid)
    X_test_flat = flatten_images(X_test)

    pca = fit_pca(X_train_flat)
    X_train_pca = transform_pca(pca, X_train_flat)
    X_valid_pca = transform_pca(pca, X_valid_flat)
    X_test_pca = transform_pca(pca, X_test_flat)

    return DatasetBundle(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        X_train_pca=X_train_pca,
        X_valid_pca=X_valid_pca,
        X_test_pca=X_test_pca,
        pca=pca,
        num_classes=len(np.unique(y_train)),
    )
