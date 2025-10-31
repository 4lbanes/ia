from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class PerceptronConfig:
    """Configuration container for the Perceptron model."""

    learning_rate: float = 1e-3
    max_epochs: int = 1000
    random_seed: int = 42
    plot_training: bool = True


class Perceptron:
    """Implementation of the classical Rosenblatt perceptron for binary classification."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        config: Optional[PerceptronConfig] = None,
    ) -> None:
        self.config = config or PerceptronConfig()

        self.X_train, self.y_train = self._prepare_inputs(X_train, y_train)
        self.num_features = self.X_train.shape[0] - 1  # subtract bias term

        self._rng = np.random.default_rng(self.config.random_seed)
        self.weights = self._rng.uniform(
            low=-0.5, high=0.5, size=(self.num_features + 1, 1)
        )

        self.training_history = {
            "epochs": 0,
            "converged": False,
            "errors_per_epoch": [],
        }
        self.weights_history = []

        if self.config.plot_training:
            self.fig, self.ax = plt.subplots()

    def _prepare_inputs(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate inputs, ensure correct shapes and append bias row."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

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

    @staticmethod
    def activation_function(value: float) -> int:
        """Binary step activation that returns either -1 or 1."""
        return 1 if value >= 0 else -1

    def fit(
        self,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train the perceptron using the perceptron learning rule."""
        self.training_history = {
            "epochs": 0,
            "converged": False,
            "errors_per_epoch": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }
        self.weights_history = [self.weights.copy()]

        learning_rate = self.config.learning_rate
        max_epochs = self.config.max_epochs

        n_samples = self.X_train.shape[1]

        for epoch in range(1, max_epochs + 1):
            errors = 0

            for i in range(n_samples):
                x_i = self.X_train[:, i : i + 1]  # column vector
                y_i = self.y_train[i]

                linear_output = float(self.weights.T @ x_i)
                prediction = self.activation_function(linear_output)

                if prediction != y_i:
                    update = learning_rate * (y_i - prediction)
                    self.weights += update * x_i
                    errors += 1

            self.weights_history.append(self.weights.copy())
            self.training_history["errors_per_epoch"].append(errors)
            train_accuracy = 1.0 - (errors / n_samples)
            self.training_history["train_accuracy"].append(train_accuracy)

            if X_val is not None and y_val is not None:
                val_predictions = self.predict(X_val)
                val_accuracy = float(np.mean(val_predictions == np.ravel(y_val)))
                self.training_history["val_accuracy"].append(val_accuracy)
            else:
                self.training_history["val_accuracy"].append(float("nan"))

            if errors == 0:
                self.training_history["converged"] = True
                break

        # Store bookkeeping information expected by the tests
        self.training_history["epochs_run"] = len(self.training_history["errors_per_epoch"])
        self.training_history["epochs"] = len(self.weights_history)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for the provided samples."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("Input data must be 2D with shape (n_features, n_samples)")

        bias_row = -np.ones((1, X.shape[1]), dtype=float)
        X_with_bias = np.vstack((bias_row, X))

        linear_output = (self.weights.T @ X_with_bias).flatten()
        activation = np.where(linear_output >= 0, 1, -1)

        return activation

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
