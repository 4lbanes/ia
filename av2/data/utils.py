from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class NormalisationStats:
    mean: np.ndarray
    std: np.ndarray


def standardize(X: np.ndarray) -> Tuple[np.ndarray, NormalisationStats]:
    """Apply z-score standardisation feature-wise."""
    X = np.asarray(X, dtype=float)
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    std_corrected = np.where(std == 0, 1.0, std)
    normalized = (X - mean) / std_corrected
    return normalized, NormalisationStats(mean=mean, std=std_corrected)


def min_max_scale(X: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Scale features to [0, 1]."""
    X = np.asarray(X, dtype=float)
    min_val = np.min(X, axis=1, keepdims=True)
    max_val = np.max(X, axis=1, keepdims=True)
    denom = np.where(max_val - min_val == 0, 1.0, max_val - min_val)
    scaled = (X - min_val) / denom
    return scaled, (min_val, max_val)


def one_hot_encode(labels: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """Transform integer labels into one-hot encoded matrix (classes x samples)."""
    labels = np.asarray(labels, dtype=int).ravel()
    if num_classes is None:
        num_classes = int(labels.max() + 1)

    one_hot = np.zeros((num_classes, labels.size), dtype=float)
    one_hot[labels, np.arange(labels.size)] = 1.0
    return one_hot


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Randomly split arrays into train and test subsets."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X must be 2D with shape (n_features, n_samples)")

    if y.ndim not in (1, 2):
        raise ValueError("y must be 1D or 2D.")

    n_samples = X.shape[1]
    indices = np.arange(n_samples)

    rng = np.random.default_rng(random_seed)
    rng.shuffle(indices)

    test_count = int(np.floor(test_size * n_samples))
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    X_train = X[:, train_idx]
    X_test = X[:, test_idx]

    if y.ndim == 1:
        y_train = y[train_idx]
        y_test = y[test_idx]
    else:
        y_train = y[:, train_idx]
        y_test = y[:, test_idx]

    return X_train, X_test, y_train, y_test

