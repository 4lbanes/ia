from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class RBFConfig:
    """Configuration options for the Radial Basis Function network."""

    num_centers: int = 10
    max_epochs: int = 300
    learning_rate: float = 1e-2
    random_seed: int = 42
    tolerance: float = 1e-5
    spread: Optional[float] = None
    kmeans_max_iter: int = 100
    l2_lambda: float = 0.0


class RBFNetwork:
    """Radial Basis Function neural network with Gaussian bases."""

    def __init__(self, config: Optional[RBFConfig] = None) -> None:
        self.config = config or RBFConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

        self.centers: Optional[np.ndarray] = None
        self.spread: Optional[float] = None
        self.weights: Optional[np.ndarray] = None

        self.training_history: Dict[str, list] = {
            "loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }
        self.output_mode: str = "zero_one"

    def _kmeans(self, X: np.ndarray) -> np.ndarray:
        """Lightweight k-means implementation to select RBF centers."""
        n_samples = X.shape[1]
        num_centers = self.config.num_centers

        if num_centers > n_samples:
            raise ValueError("Number of centers cannot exceed number of samples.")

        centers = X[:, self._rng.choice(n_samples, size=num_centers, replace=False)]

        for _ in range(self.config.kmeans_max_iter):
            distances = self._pairwise_squared_distances(X, centers)
            assignments = np.argmin(distances, axis=1)

            new_centers = np.zeros_like(centers)
            for idx in range(num_centers):
                cluster_points = X[:, assignments == idx]
                if cluster_points.size == 0:
                    new_centers[:, idx] = centers[:, idx]
                else:
                    new_centers[:, idx] = cluster_points.mean(axis=1)

            if np.allclose(new_centers, centers):
                break
            centers = new_centers

        return centers

    @staticmethod
    def _pairwise_squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Compute squared Euclidean distances between samples and centers."""
        # X shape: (n_features, n_samples), centers: (n_features, num_centers)
        diff = X[:, :, None] - centers[:, None, :]  # (n_features, n_samples, num_centers)
        squared = np.sum(diff**2, axis=0)  # (n_samples, num_centers)
        return squared

    def _compute_spread(self, centers: np.ndarray) -> float:
        """Heuristic spread based on mean pairwise center distance."""
        if self.config.spread is not None:
            return self.config.spread

        num_centers = centers.shape[1]
        if num_centers < 2:
            return 1.0

        dists = self._pairwise_squared_distances(centers, centers)
        upper = dists[np.triu_indices(num_centers, k=1)]
        mean_distance = np.sqrt(np.mean(upper))
        return mean_distance if mean_distance > 0 else 1.0

    def _rbf_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply RBF transformation to inputs."""
        if self.centers is None or self.spread is None:
            raise RuntimeError("Model must be fitted before calling _rbf_transform.")

        squared_distances = self._pairwise_squared_distances(X, self.centers)
        phi = np.exp(-squared_distances / (2.0 * self.spread**2))
        return phi.T  # shape: (num_centers, n_samples)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train the RBF network using gradient descent on the output layer."""
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)

        if X_train.ndim != 2:
            raise ValueError("X_train must be 2D with shape (n_features, n_samples)")
        if y_train.ndim != 2:
            raise ValueError("y_train must be 2D with shape (n_outputs, n_samples)")

        self.centers = self._kmeans(X_train)
        self.spread = self._compute_spread(self.centers)

        phi = self._rbf_transform(X_train)
        phi_bias = np.vstack((np.ones((1, phi.shape[1])), phi))

        n_outputs = y_train.shape[0]
        self.weights = self._rng.uniform(
            low=-0.5, high=0.5, size=(n_outputs, phi_bias.shape[0])
        )

        unique_values = np.unique(y_train)
        if y_train.shape[0] == 1:
            if np.all(np.isin(unique_values, [-1, 1])):
                self.output_mode = "minus_plus"
            else:
                self.output_mode = "zero_one"
        else:
            self.output_mode = "one_hot"

        self.training_history = {"loss": [], "train_accuracy": [], "val_accuracy": []}

        prev_loss = np.inf
        lr = self.config.learning_rate

        for epoch in range(1, self.config.max_epochs + 1):
            outputs = self.weights @ phi_bias
            errors = outputs - y_train

            loss = 0.5 * np.mean(errors**2)
            self.training_history["loss"].append(loss)

            grad = (errors @ phi_bias.T) / phi_bias.shape[1]
            if self.config.l2_lambda > 0:
                grad += self.config.l2_lambda * self.weights

            self.weights -= lr * grad

            train_accuracy = self._compute_accuracy(y_train, outputs)
            self.training_history["train_accuracy"].append(train_accuracy)

            if X_val is not None and y_val is not None:
                y_val_pred = self.predict_proba(X_val)
                val_accuracy = self._compute_accuracy(y_val, y_val_pred)
                self.training_history["val_accuracy"].append(val_accuracy)
            else:
                self.training_history["val_accuracy"].append(float("nan"))

            if np.abs(prev_loss - loss) < self.config.tolerance:
                break
            prev_loss = loss

    def _compute_accuracy(self, y_true: np.ndarray, outputs: np.ndarray) -> float:
        """Helper to compute accuracy from raw outputs."""
        if y_true.shape[0] == 1:
            if self.output_mode == "minus_plus":
                preds = np.where(outputs.flatten() >= 0, 1, -1)
                return float(np.mean(preds == y_true.flatten()))
            preds = np.where(outputs.flatten() >= 0.5, 1, 0)
            return float(np.mean(preds == y_true.flatten()))

        pred_labels = np.argmax(outputs, axis=0)
        true_labels = np.argmax(y_true, axis=0)
        return float(np.mean(pred_labels == true_labels))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return raw outputs of the network."""
        if self.centers is None or self.weights is None:
            raise RuntimeError("The model must be fitted before calling predict_proba.")

        X = np.asarray(X, dtype=float)
        phi = self._rbf_transform(X)
        phi_bias = np.vstack((np.ones((1, phi.shape[1])), phi))
        return self.weights @ phi_bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return discrete predictions."""
        outputs = self.predict_proba(X)

        if outputs.shape[0] == 1:
            if self.output_mode == "minus_plus":
                return np.where(outputs.flatten() >= 0, 1, -1)
            return np.where(outputs.flatten() >= 0.5, 1, 0)

        return np.argmax(outputs, axis=0)
