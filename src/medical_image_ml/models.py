from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.ensemble import RandomForestClassifier


def build_random_forest(
    n_estimators: int = 250,
    max_depth: int | None = 20,
    max_features: str = "sqrt",
    random_state: int = 42,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )


def build_mlp(
    input_dim: int,
    num_classes: int,
    units: int = 256,
    activation: str = "relu",
    dropout: float = 0.2,
    hidden_layers: int = 2,
    learning_rate: float = 0.1,
) -> keras.Model:
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    for _ in range(hidden_layers):
        model.add(keras.layers.Dense(units, activation=activation))
        model.add(keras.layers.Dropout(rate=dropout))
    model.add(keras.layers.Dense(num_classes, activation="softmax", dtype="float32"))
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn(
    input_shape: tuple[int, int, int],
    num_classes: int,
    filters: tuple[int, int] = (32, 64),
    kernels: tuple[int, int] = (3, 5),
    dense_units: int = 64,
    dropout_conv: float = 0.3,
    dropout_dense: float = 0.3,
    learning_rate: float = 1e-3,
    l2: float = 1e-4,
) -> keras.Model:
    l2_reg = regularizers.l2(l2)
    model = keras.Sequential(
        [
            layers.Conv2D(
                filters[0],
                kernels[0],
                padding="same",
                activation="relu",
                kernel_regularizer=l2_reg,
                input_shape=input_shape,
            ),
            layers.BatchNormalization(),
            layers.Conv2D(
                filters[1],
                kernels[1],
                padding="same",
                activation="relu",
                kernel_regularizer=l2_reg,
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(dropout_conv),
            layers.GlobalAveragePooling2D(),
            layers.Dense(dense_units, activation="relu", kernel_regularizer=l2_reg),
            layers.Dropout(dropout_dense),
            layers.Dense(num_classes, activation="softmax", dtype="float32"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
