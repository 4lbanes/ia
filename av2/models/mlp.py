from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _sigmoid_derivative(output: np.ndarray) -> np.ndarray:
    return output * (1.0 - output)


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _tanh_derivative(output: np.ndarray) -> np.ndarray:
    return 1.0 - output**2


ACTIVATIONS: Dict[str, Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]] = {
    "sigmoid": (_sigmoid, _sigmoid_derivative),
    "tanh": (_tanh, _tanh_derivative),
}


@dataclass
class MLPConfig:
    """Configuration parameters for the MLP model."""

    hidden_layers: List[int] = field(default_factory=lambda: [8])
    learning_rate: float = 1e-2
    max_epochs: int = 500
    random_seed: int = 42
    activation: str = "tanh"
    tolerance: float = 1e-5
    batch_size: Optional[int] = None  # None -> full batch
    l2_lambda: float = 0.0

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation '{self.activation}'. Use one of {list(ACTIVATIONS)}.")


class MLP:
    """Simple fully connected neural network with configurable topology."""

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        config: Optional[MLPConfig] = None,
    ) -> None:
        self.config = config or MLPConfig()
        self.n_features = n_features
        self.n_outputs = n_outputs

        layers = [n_features] + list(self.config.hidden_layers) + [n_outputs]
        self._rng = np.random.default_rng(self.config.random_seed)

        # Heuristic weight init: scaled uniform
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for fan_in, fan_out in zip(layers[:-1], layers[1:]):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.weights.append(self._rng.uniform(-limit, limit, size=(fan_out, fan_in)))
            self.biases.append(np.zeros((fan_out, 1)))

        self.activation, self.activation_derivative = ACTIVATIONS[self.config.activation]

        self.training_history: Dict[str, List[float]] = {
            "loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Compute forward pass returning linear combinations and activations."""
        z_values: List[np.ndarray] = []
        activations: List[np.ndarray] = [X]

        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            Z = W @ activations[-1] + b
            A = self.activation(Z)
            z_values.append(Z)
            activations.append(A)

        # Output layer uses same activation as mandated
        Z_out = self.weights[-1] @ activations[-1] + self.biases[-1]
        A_out = self.activation(Z_out)
        z_values.append(Z_out)
        activations.append(A_out)

        return z_values, activations

    def _compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute accuracy for binary or multi-class outputs."""
        if y_true.ndim == 1 or y_true.shape[0] == 1:
            y_true_labels = (np.ravel(y_true) >= 0.5).astype(int)
            y_pred_labels = (np.ravel(y_pred) >= 0.5).astype(int)
            return float(np.mean(y_true_labels == y_pred_labels))

        true_labels = np.argmax(y_true, axis=0)
        pred_labels = np.argmax(y_pred, axis=0)
        return float(np.mean(true_labels == pred_labels))

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train the MLP using batch or mini-batch gradient descent."""
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)

        if X_train.ndim != 2:
            raise ValueError("X_train must be 2D with shape (n_features, n_samples)")
        if y_train.ndim != 2:
            raise ValueError("y_train must be 2D with shape (n_outputs, n_samples)")

        n_samples = X_train.shape[1]
        batch_size = self.config.batch_size or n_samples

        self.training_history = {"loss": [], "train_accuracy": [], "val_accuracy": []}

        prev_loss = np.inf

        for epoch in range(1, self.config.max_epochs + 1):
            # Shuffle indices for mini-batch training
            indices = np.arange(n_samples)
            self._rng.shuffle(indices)

            cumulative_loss = 0.0
            correct = 0
            processed_samples = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]
                X_batch = X_train[:, batch_idx]
                y_batch = y_train[:, batch_idx]

                z_values, activations = self._forward(X_batch)
                y_pred = activations[-1]

                errors = y_pred - y_batch
                batch_loss = 0.5 * np.mean(errors**2)
                cumulative_loss += batch_loss * (end - start)

                delta = errors * self.activation_derivative(y_pred)
                deltas: List[np.ndarray] = [delta]

                for layer in range(len(self.weights) - 2, -1, -1):
                    delta = (
                        self.weights[layer + 1].T @ deltas[0]
                    ) * self.activation_derivative(activations[layer + 1])
                    deltas.insert(0, delta)

                for idx_layer, (W, b, delta_l, activation_prev) in enumerate(
                    zip(self.weights, self.biases, deltas, activations[:-1])
                ):
                    grad_W = (delta_l @ activation_prev.T) / (end - start)
                    grad_b = np.mean(delta_l, axis=1, keepdims=True)

                    if self.config.l2_lambda > 0:
                        grad_W += (self.config.l2_lambda / (end - start)) * W

                    self.weights[idx_layer] -= self.config.learning_rate * grad_W
                    self.biases[idx_layer] -= self.config.learning_rate * grad_b

                batch_accuracy = self._compute_accuracy(y_batch, y_pred)
                correct += batch_accuracy * (end - start)
                processed_samples += end - start

            avg_loss = cumulative_loss / processed_samples
            train_accuracy = correct / processed_samples

            self.training_history["loss"].append(avg_loss)
            self.training_history["train_accuracy"].append(train_accuracy)

            if X_val is not None and y_val is not None:
                y_val_pred = self.predict_proba(X_val)
                val_accuracy = self._compute_accuracy(y_val, y_val_pred)
                self.training_history["val_accuracy"].append(val_accuracy)
            else:
                self.training_history["val_accuracy"].append(float("nan"))

            if np.abs(prev_loss - avg_loss) < self.config.tolerance:
                break
            prev_loss = avg_loss

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return network outputs (probabilities / activations)."""
        X = np.asarray(X, dtype=float)
        _, activations = self._forward(X)
        return activations[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return discrete class predictions."""
        outputs = self.predict_proba(X)

        if outputs.shape[0] == 1:
            return np.where(outputs.flatten() >= 0.5, 1, 0)

        return np.argmax(outputs, axis=0)

