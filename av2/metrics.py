from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


def confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """Compute a confusion matrix without relying on external libraries."""
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = list(labels)

    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    for true_label, pred_label in zip(y_true, y_pred):
        i = label_to_index[true_label]
        j = label_to_index[pred_label]
        matrix[i, j] += 1

    return matrix


def accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    return float(np.mean(y_true == y_pred))


def _binary_labels(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    positive: int,
    negative: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))

    if negative is None:
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        negatives = [label for label in unique_labels if label != positive]
        if not negatives:
            raise ValueError("Unable to determine negative class.")
        negative = negatives[0]

    return y_true, y_pred, positive, negative


def precision_score(
    y_true: Sequence[int], y_pred: Sequence[int], positive: int = 1, negative: Optional[int] = None
) -> float:
    y_true, y_pred, positive, negative = _binary_labels(y_true, y_pred, positive, negative)
    cm = confusion_matrix(y_true, y_pred, labels=[positive, negative])
    tp = cm[0, 0]
    fp = cm[1, 0]
    denominator = tp + fp
    return float(tp / denominator) if denominator else 0.0


def recall_score(
    y_true: Sequence[int], y_pred: Sequence[int], positive: int = 1, negative: Optional[int] = None
) -> float:
    y_true, y_pred, positive, negative = _binary_labels(y_true, y_pred, positive, negative)
    cm = confusion_matrix(y_true, y_pred, labels=[positive, negative])
    tp = cm[0, 0]
    fn = cm[0, 1]
    denominator = tp + fn
    return float(tp / denominator) if denominator else 0.0


def specificity_score(
    y_true: Sequence[int], y_pred: Sequence[int], positive: int = 1, negative: Optional[int] = None
) -> float:
    y_true, y_pred, positive, negative = _binary_labels(y_true, y_pred, positive, negative)
    cm = confusion_matrix(y_true, y_pred, labels=[positive, negative])
    tn = cm[1, 1]
    fp = cm[1, 0]
    denominator = tn + fp
    return float(tn / denominator) if denominator else 0.0


def f1_score(
    y_true: Sequence[int], y_pred: Sequence[int], positive: int = 1, negative: Optional[int] = None
) -> float:
    precision = precision_score(y_true, y_pred, positive, negative)
    recall = recall_score(y_true, y_pred, positive, negative)
    denominator = precision + recall
    return float((2 * precision * recall) / denominator) if denominator else 0.0


@dataclass
class MetricSummary:
    mean: float
    std: float
    maximum: float
    minimum: float


def summarise_metric(values: Sequence[float]) -> MetricSummary:
    data = np.asarray(list(values), dtype=float)
    return MetricSummary(
        mean=float(np.mean(data)),
        std=float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
        maximum=float(np.max(data)),
        minimum=float(np.min(data)),
    )


def summarise_metrics(results: Dict[str, Sequence[float]]) -> Dict[str, MetricSummary]:
    return {metric: summarise_metric(values) for metric, values in results.items()}


def compute_binary_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    positive: int = 1,
    negative: Optional[int] = None,
) -> Dict[str, float]:
    """Return binary metrics: accuracy, precision, recall, specificity, f1."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, positive, negative)
    rec = recall_score(y_true, y_pred, positive, negative)
    spec = specificity_score(y_true, y_pred, positive, negative)
    f1 = f1_score(y_true, y_pred, positive, negative)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1_score": f1,
    }
