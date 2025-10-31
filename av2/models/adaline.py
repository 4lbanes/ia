from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class AdalineConfig:
    """Configuration parameters for the ADALINE model."""

    learning_rate: float = 1e-2
    max_epochs: int = 1000
    random_seed: int = 42
    tolerance: float = 1e-5
    plot_training: bool = True


class Adaline:
    """ADAptive LInear NEuron (ADALINE) for binary classification."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        config: Optional[AdalineConfig] = None,
    ) -> None:
        self.config = config or AdalineConfig()

        self.X_train, self.y_train = self._prepare_inputs(X_train, y_train)
        self.num_features = self.X_train.shape[0] - 1

        self._rng = np.random.default_rng(self.config.random_seed)
        self.weights = self._rng.uniform(
            low=-0.5, high=0.5, size=(self.num_features + 1, 1)
        )

        self.training_history = {
            "epochs": 0,
            "converged": False,
            "mse_per_epoch": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }
        self.weights_history = []

        if self.config.plot_training:
            self.fig, self.ax = plt.subplots()

    def _prepare_inputs(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate inputs, ensure correct shapes and append bias row."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X_train must be a 2D array with shape (n_features, n_samples)")

        if y.ndim not in (1, 2):
            raise ValueError("y_train must be a 1D or 2D array")

        y = np.ravel(y)

        n_samples = X.shape[1]
        if y.shape[0] != n_samples:
            raise ValueError("Number of samples in X_train and y_train must match")

        bias_row = -np.ones((1, n_samples), dtype=float)
        X_with_bias = np.vstack((bias_row, X))

        return X_with_bias, y

    def fit(self, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Train the ADALINE using batch gradient descent."""
        lr = self.config.learning_rate
        max_epochs = self.config.max_epochs
        tolerance = self.config.tolerance

        self.weights_history = [self.weights.copy()]
        previous_mse = np.inf

        val_bias = None
        y_val_flat = None
        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=float)
            y_val = np.asarray(y_val)
            if X_val.ndim != 2:
                raise ValueError("X_val must be 2D with shape (n_features, n_samples)")
            if y_val.ndim not in (1, 2):
                raise ValueError("y_val must be 1D or 2D")
            y_val_flat = np.ravel(y_val)
            if X_val.shape[1] != y_val_flat.shape[0]:
                raise ValueError("Validation set: X_val and y_val must share the number of samples.")
            val_bias = -np.ones((1, X_val.shape[1]), dtype=float)
            X_val = np.vstack((val_bias, X_val))

        for epoch in range(1, max_epochs + 1):
            net_input = (self.weights.T @ self.X_train).flatten()
            errors = self.y_train - net_input

            gradient = -(2.0 / self.X_train.shape[1]) * (self.X_train @ errors[:, None])
            self.weights -= lr * gradient

            mse = np.mean(errors ** 2)

            self.training_history["mse_per_epoch"].append(mse)
            self.weights_history.append(self.weights.copy())

            train_outputs = (self.weights.T @ self.X_train).flatten()
            train_predictions = np.where(train_outputs >= 0, 1, -1)
            train_accuracy = float(np.mean(train_predictions == self.y_train))
            self.training_history["train_accuracy"].append(train_accuracy)

            if X_val is not None and y_val is not None and val_bias is not None:
                val_outputs = (self.weights.T @ X_val).flatten()
                val_predictions = np.where(val_outputs >= 0, 1, -1)
                self.training_history["val_accuracy"].append(
                    float(np.mean(val_predictions == y_val_flat))
                )
            else:
                self.training_history["val_accuracy"].append(float("nan"))

            if np.abs(previous_mse - mse) < tolerance:
                self.training_history["converged"] = True
                break

            previous_mse = mse

        self.training_history["epochs_run"] = len(self.training_history["mse_per_epoch"])
        self.training_history["epochs"] = len(self.weights_history)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for the provided samples."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("Input data must be 2D with shape (n_features, n_samples)")

        bias_row = -np.ones((1, X.shape[1]), dtype=float)
        X_with_bias = np.vstack((bias_row, X))

        net_input = (self.weights.T @ X_with_bias).flatten()
        return np.where(net_input >= 0, 1, -1)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return the raw linear output without applying the activation."""
        X = np.asarray(X, dtype=float)

        bias_row = -np.ones((1, X.shape[1]), dtype=float)
        X_with_bias = np.vstack((bias_row, X))

        return (self.weights.T @ X_with_bias).flatten()

    def _compute_decision_boundary(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute coordinates for the decision boundary (2D problems only)."""
        if self.num_features != 2:
            raise ValueError("Decision boundary can only be computed for 2D data.")

        w0, w1, w2 = self.weights.flatten()

        feature_data = self.X_train[1:, :]
        x_min = feature_data[0, :].min() - 1.0
        x_max = feature_data[0, :].max() + 1.0
        x1 = np.linspace(x_min, x_max, 100)

        denominator = w2 if not np.isclose(w2, 0.0) else np.finfo(float).eps
        x2 = -(w0 + w1 * x1) / denominator
        x2 = np.nan_to_num(x2, nan=np.nanmedian(x2), posinf=np.nanmedian(x2), neginf=np.nanmedian(x2))

        return x1, x2
