from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_score, recall_score
from tensorflow import keras


class PrecisionRecallCallback(keras.callbacks.Callback):
    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.precisions: list[float] = []
        self.recalls: list[float] = []
        self.train_precisions: list[float] = []
        self.train_recalls: list[float] = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred_val_probs = self.model.predict(self.X_val, verbose=0)
        y_pred_val = np.argmax(y_pred_val_probs, axis=1)
        self.precisions.append(precision_score(self.y_val, y_pred_val, average="macro"))
        self.recalls.append(recall_score(self.y_val, y_pred_val, average="macro"))

        y_pred_train_probs = self.model.predict(self.X_train, verbose=0)
        y_pred_train = np.argmax(y_pred_train_probs, axis=1)
        self.train_precisions.append(
            precision_score(self.y_train, y_pred_train, average="macro")
        )
        self.train_recalls.append(recall_score(self.y_train, y_pred_train, average="macro"))
