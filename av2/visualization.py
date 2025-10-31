from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _ensure_directory(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        # In read-only environments the directory may already exist.
        pass


def plot_scatter_2d(
    X: np.ndarray,
    y: np.ndarray,
    *,
    path: Optional[str | Path] = None,
    title: str = "Distribuição do Conjunto Spiral",
    xlabel: str = "Atributo 1",
    ylabel: str = "Atributo 2",
) -> plt.Figure:
    """Scatter plot helper for 2D datasets."""
    X = np.asarray(X)
    y = np.asarray(y)

    fig, ax = plt.subplots()
    scatter = ax.scatter(X[0, :], X[1, :], c=y, cmap="coolwarm", edgecolors="k")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    legend = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    if path is not None:
        output_path = Path(path)
        _ensure_directory(output_path)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_confusion_heatmap(
    matrix: np.ndarray,
    labels: Iterable[str],
    *,
    path: Optional[str | Path] = None,
    title: str = "Matriz de Confusão",
    annot: bool = True,
    fmt: str = "d",
) -> plt.Figure:
    """Plot confusion matrix using seaborn heatmap."""
    matrix = np.asarray(matrix)
    labels = list(labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=annot, fmt=fmt, cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(title)

    if path is not None:
        output_path = Path(path)
        _ensure_directory(output_path)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_learning_curve(
    train_values: List[float],
    val_values: Optional[List[float]] = None,
    *,
    path: Optional[str | Path] = None,
    title: str = "Curva de Aprendizagem",
    ylabel: str = "Acurácia",
) -> plt.Figure:
    """Plot learning curve with optional validation series."""
    epochs = np.arange(1, len(train_values) + 1)

    fig, ax = plt.subplots()
    ax.plot(epochs, train_values, label="Treinamento", marker="o")
    if val_values is not None and len(val_values) == len(train_values):
        ax.plot(epochs, val_values, label="Validação", marker="s")
    ax.set_xlabel("Épocas")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()

    if path is not None:
        output_path = Path(path)
        _ensure_directory(output_path)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_metric_boxplot(
    metrics: Dict[str, List[float]],
    *,
    path: Optional[str | Path] = None,
    title: str = "Distribuição de Métricas",
    ylabel: str = "Valor",
) -> plt.Figure:
    """Generate a boxplot (or violin plot) for metrics gathered across modelos."""
    categories = []
    data = []
    for model_name, values in metrics.items():
        categories.extend([model_name] * len(values))
        data.extend(values)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=categories, y=data, ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Modelo")

    if path is not None:
        output_path = Path(path)
        _ensure_directory(output_path)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig

