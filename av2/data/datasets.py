from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .utils import NormalisationStats, standardize


@dataclass
class Dataset:
    features: np.ndarray
    labels: np.ndarray
    normalisation: Optional[NormalisationStats] = None
    metadata: Dict[str, object] = field(default_factory=dict)


def load_spiral_dataset(
    csv_path: str | Path,
    *,
    normalize: bool = True,
    has_header: bool = True,
    label_mode: str = "minus_plus",
) -> Dataset:
    """Load the spiral dataset used in stage 1 of the assignment."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Spiral dataset not found: {path}")

    raw = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=1 if has_header else 0,
        dtype=float,
    )

    if raw.ndim == 1:
        raw = raw[None, :]

    if raw.shape[1] < 3:
        raise ValueError("Expected at least three columns (x1, x2, label).")

    X = raw[:, :2].T  # shape: (2, n_samples)
    y = raw[:, 2].astype(int)

    if label_mode == "minus_plus" and np.array_equal(np.unique(y), np.array([0, 1])):
        y = np.where(y == 0, -1, 1)
    elif label_mode == "zero_one" and np.array_equal(np.unique(y), np.array([-1, 1])):
        y = np.where(y == -1, 0, 1)

    norm_stats: Optional[NormalisationStats] = None
    if normalize:
        X, norm_stats = standardize(X)

    metadata = {
        "path": str(path),
        "n_samples": X.shape[1],
        "feature_dim": X.shape[0],
        "label_mode": label_mode,
        "labels": np.unique(y).tolist(),
    }

    return Dataset(features=X, labels=y, normalisation=norm_stats, metadata=metadata)

