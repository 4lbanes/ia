from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .datasets import Dataset
from .utils import min_max_scale


@dataclass
class RecFacMetadata:
    label_to_person: Dict[int, str] = field(default_factory=dict)
    person_to_label: Dict[str, int] = field(default_factory=dict)
    original_shape: Tuple[int, int] = (120, 128)
    resized_shape: Tuple[int, int] = (40, 40)


def load_recfac_dataset(
    root_dir: str | Path,
    *,
    image_size: Tuple[int, int] = (40, 40),
    normalize: bool = True,
    flatten: bool = True,
    accepted_extensions: Iterable[str] = (".pgm", ".jpg", ".jpeg", ".png"),
) -> Dataset:
    """Load the RecFac face recognition dataset."""
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"RecFac dataset directory not found: {root_path}")

    images: List[np.ndarray] = []
    labels: List[int] = []
    label_to_person: Dict[int, str] = {}
    person_to_label: Dict[str, int] = {}

    extensions = tuple(ext.lower() for ext in accepted_extensions)

    original_shape: Optional[Tuple[int, int]] = None

    for person_idx, person_dir in enumerate(sorted(p for p in root_path.iterdir() if p.is_dir())):
        person_to_label[person_dir.name] = person_idx
        label_to_person[person_idx] = person_dir.name

        for image_path in sorted(
            p for p in person_dir.rglob("*") if p.suffix.lower() in extensions
        ):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            original_shape = image.shape[:2]
            if image_size is not None:
                image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)

            if normalize:
                image = image.astype(float) / 255.0

            images.append(image)
            labels.append(person_idx)

    if not images:
        raise ValueError(f"No images found in {root_path}. Check the directory structure.")

    data = np.stack(images, axis=0)
    y = np.asarray(labels, dtype=int)

    if flatten:
        data = data.reshape(data.shape[0], -1)

    X = data.T  # (n_features, n_samples)

    if not normalize:
        X, (min_val, max_val) = min_max_scale(X)
        normalisation = None
    else:
        min_val = np.min(X)
        max_val = np.max(X)
        normalisation = None

    metadata = RecFacMetadata(
        label_to_person=label_to_person,
        person_to_label=person_to_label,
        original_shape=original_shape,
        resized_shape=image_size,
    )

    dataset_metadata = {
        "path": str(root_path),
        "n_samples": X.shape[1],
        "feature_dim": X.shape[0],
        "labels": list(label_to_person.values()),
        "normalised": normalize,
        "min_pixel_value": float(min_val),
        "max_pixel_value": float(max_val),
        "recfac": metadata,
    }

    return Dataset(features=X, labels=y, normalisation=None, metadata=dataset_metadata)
