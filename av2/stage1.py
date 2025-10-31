from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional

import numpy as np
from tqdm import trange

from .data.datasets import Dataset, load_spiral_dataset
from .data.utils import train_test_split
from .metrics import MetricSummary, compute_binary_classification_metrics, confusion_matrix, summarise_metrics
from .models.adaline import Adaline, AdalineConfig
from .models.mlp import MLP, MLPConfig
from .models.rbf import RBFNetwork, RBFConfig
from .models.simple_perceptron import Perceptron, PerceptronConfig


logger = logging.getLogger(__name__)


def _to_zero_one(y: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(y) == -1, 0, 1)


def _to_minus_plus(y: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(y) == 0, -1, 1)


@dataclass
class Stage1Config:
    dataset_path: str
    test_size: float = 0.2
    monte_carlo_runs: int = 500
    random_seed: int = 42
    perceptron_config: PerceptronConfig = field(
        default_factory=lambda: PerceptronConfig(plot_training=False)
    )
    adaline_config: AdalineConfig = field(
        default_factory=lambda: AdalineConfig(plot_training=False)
    )
    mlp_config: MLPConfig = field(
        default_factory=lambda: MLPConfig(hidden_layers=[12, 8], activation="tanh")
    )
    rbf_config: RBFConfig = field(
        default_factory=lambda: RBFConfig(num_centers=25, max_epochs=300)
    )
    mlp_topologies: Optional[Dict[str, MLPConfig]] = None
    rbf_topologies: Optional[Dict[str, RBFConfig]] = None


@dataclass
class ModelRunRecord:
    y_true: np.ndarray
    y_pred: np.ndarray
    confusion: np.ndarray
    history: Dict[str, List[float]]


@dataclass
class ModelMonteCarloResult:
    metrics: Dict[str, np.ndarray]
    records: List[ModelRunRecord]
    metric_summaries: Dict[str, MetricSummary]
    best_indices: Dict[str, int]
    worst_indices: Dict[str, int]

    def confusion_for_metric(self, metric: str, best: bool = True) -> np.ndarray:
        idx = self.best_indices[metric] if best else self.worst_indices[metric]
        return self.records[idx].confusion

    def learning_curve(self, metric: str, best: bool = True) -> Dict[str, List[float]]:
        idx = self.best_indices[metric] if best else self.worst_indices[metric]
        return self.records[idx].history


@dataclass
class Stage1Results:
    dataset: Dataset
    config: Stage1Config
    model_results: Dict[str, ModelMonteCarloResult]


def _prepare_model_histories(model) -> Dict[str, List[float]]:
    history = getattr(model, "training_history", {})
    return {
        key: list(values)
        for key, values in history.items()
        if isinstance(values, (list, tuple))
    }


def run_stage1_monte_carlo(config: Stage1Config) -> Stage1Results:
    """Execute Monte Carlo evaluation for Stage 1."""
    logger.info("Carregando dataset Spiral de %s", config.dataset_path)
    dataset = load_spiral_dataset(config.dataset_path, normalize=True, label_mode="minus_plus")
    X = dataset.features
    y = dataset.labels

    rng = np.random.default_rng(config.random_seed)
    runs = config.monte_carlo_runs

    logger.info("Iniciando simulação Stage 1 com %d rodadas (seed base=%d)", runs, config.random_seed)

    models = ["perceptron", "adaline", "mlp", "rbf"]
    metrics_template = {metric: np.zeros(runs) for metric in ["accuracy", "precision", "recall", "specificity", "f1_score"]}

    model_metrics: Dict[str, Dict[str, np.ndarray]] = {
        name: {metric: np.zeros(runs) for metric in metrics_template} for name in models
    }
    model_records: Dict[str, List[ModelRunRecord]] = {name: [] for name in models}

    for run in trange(runs, desc="Etapa 1 - Monte Carlo", leave=False):
        seed = int(rng.integers(0, 1_000_000_000))
        logger.info("Stage 1 - Rodada %d/%d (seed=%d)", run + 1, runs, seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_seed=seed
        )

        # Perceptron
        perceptron_cfg = replace(config.perceptron_config, random_seed=seed)
        perceptron = Perceptron(X_train, y_train, perceptron_cfg)
        perceptron.fit(X_val=X_test, y_val=y_test)
        y_pred_perc = perceptron.predict(X_test)
        metrics_perc = compute_binary_classification_metrics(y_test, y_pred_perc, positive=1, negative=-1)
        for metric, value in metrics_perc.items():
            model_metrics["perceptron"][metric][run] = value
        confusion_perc = confusion_matrix(y_test, y_pred_perc, labels=[-1, 1])
        model_records["perceptron"].append(
            ModelRunRecord(
                y_true=y_test,
                y_pred=y_pred_perc,
                confusion=confusion_perc,
                history=_prepare_model_histories(perceptron),
            )
        )

        # ADALINE
        adaline_cfg = replace(config.adaline_config, random_seed=seed)
        adaline = Adaline(X_train, y_train, adaline_cfg)
        adaline.fit(X_val=X_test, y_val=y_test)
        y_pred_adaline = adaline.predict(X_test)
        metrics_ada = compute_binary_classification_metrics(y_test, y_pred_adaline, positive=1, negative=-1)
        for metric, value in metrics_ada.items():
            model_metrics["adaline"][metric][run] = value
        confusion_ada = confusion_matrix(y_test, y_pred_adaline, labels=[-1, 1])
        model_records["adaline"].append(
            ModelRunRecord(
                y_true=y_test,
                y_pred=y_pred_adaline,
                confusion=confusion_ada,
                history=_prepare_model_histories(adaline),
            )
        )

        # MLP
        mlp_cfg = replace(config.mlp_config, random_seed=seed)
        mlp = MLP(n_features=X_train.shape[0], n_outputs=1, config=mlp_cfg)
        y_train_mlp = _to_zero_one(y_train).reshape(1, -1)
        y_test_mlp = _to_zero_one(y_test).reshape(1, -1)
        mlp.fit(X_train, y_train_mlp, X_val=X_test, y_val=y_test_mlp)
        y_pred_mlp = mlp.predict(X_test)
        y_pred_mlp_pm = _to_minus_plus(y_pred_mlp)
        metrics_mlp = compute_binary_classification_metrics(y_test, y_pred_mlp_pm, positive=1, negative=-1)
        for metric, value in metrics_mlp.items():
            model_metrics["mlp"][metric][run] = value
        confusion_mlp = confusion_matrix(y_test, y_pred_mlp_pm, labels=[-1, 1])
        model_records["mlp"].append(
            ModelRunRecord(
                y_true=y_test,
                y_pred=y_pred_mlp_pm,
                confusion=confusion_mlp,
                history=_prepare_model_histories(mlp),
            )
        )

        # RBF
        rbf_cfg = replace(config.rbf_config, random_seed=seed)
        rbf = RBFNetwork(config=rbf_cfg)
        y_train_rbf = y_train_mlp
        y_test_rbf = y_test_mlp
        rbf.fit(X_train, y_train_rbf, X_val=X_test, y_val=y_test_rbf)
        y_pred_rbf = rbf.predict(X_test)
        y_pred_rbf_pm = _to_minus_plus(y_pred_rbf)
        metrics_rbf = compute_binary_classification_metrics(y_test, y_pred_rbf_pm, positive=1, negative=-1)
        for metric, value in metrics_rbf.items():
            model_metrics["rbf"][metric][run] = value
        confusion_rbf = confusion_matrix(y_test, y_pred_rbf_pm, labels=[-1, 1])
        model_records["rbf"].append(
            ModelRunRecord(
                y_true=y_test,
                y_pred=y_pred_rbf_pm,
                confusion=confusion_rbf,
                history=_prepare_model_histories(rbf),
            )
        )

    model_results: Dict[str, ModelMonteCarloResult] = {}
    for model_name in models:
        metrics_dict = model_metrics[model_name]
        records = model_records[model_name]
        summaries = summarise_metrics(metrics_dict)
        best_indices = {metric: int(np.argmax(values)) for metric, values in metrics_dict.items()}
        worst_indices = {metric: int(np.argmin(values)) for metric, values in metrics_dict.items()}
        model_results[model_name] = ModelMonteCarloResult(
            metrics=metrics_dict,
            records=records,
            metric_summaries=summaries,
            best_indices=best_indices,
            worst_indices=worst_indices,
        )

    logger.info("Concluídas as %d rodadas da Stage 1", runs)

    return Stage1Results(dataset=dataset, config=config, model_results=model_results)


def analyse_stage1_topologies(
    dataset_path: str,
    mlp_topologies: Dict[str, MLPConfig],
    rbf_topologies: Dict[str, RBFConfig],
    *,
    test_size: float = 0.2,
    random_seed: int = 42,
    repeats: int = 5,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Evaluate underfitting/overfitting scenarios for MLP and RBF topologies."""
    dataset = load_spiral_dataset(dataset_path, normalize=True, label_mode="minus_plus")
    X = dataset.features
    y = dataset.labels

    rng = np.random.default_rng(random_seed)
    results: Dict[str, Dict[str, Dict[str, float]]] = {"mlp": {}, "rbf": {}}

    for name, mlp_cfg in mlp_topologies.items():
        metrics_accum = []
        for _ in range(repeats):
            seed = int(rng.integers(0, 1_000_000_000))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_seed=seed)
            mlp = MLP(X_train.shape[0], 1, replace(mlp_cfg, random_seed=seed))
            y_train_mlp = _to_zero_one(y_train).reshape(1, -1)
            y_test_mlp = _to_zero_one(y_test).reshape(1, -1)
            mlp.fit(X_train, y_train_mlp, X_val=X_test, y_val=y_test_mlp)
            y_pred = _to_minus_plus(mlp.predict(X_test))
            metrics_accum.append(
                compute_binary_classification_metrics(y_test, y_pred, positive=1, negative=-1)
            )
        aggregated = {key: float(np.mean([m[key] for m in metrics_accum])) for key in metrics_accum[0]}
        results["mlp"][name] = aggregated

    for name, rbf_cfg in rbf_topologies.items():
        metrics_accum = []
        for _ in range(repeats):
            seed = int(rng.integers(0, 1_000_000_000))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_seed=seed)
            rbf = RBFNetwork(config=replace(rbf_cfg, random_seed=seed))
            y_train_rbf = _to_zero_one(y_train).reshape(1, -1)
            y_test_rbf = _to_zero_one(y_test).reshape(1, -1)
            rbf.fit(X_train, y_train_rbf, X_val=X_test, y_val=y_test_rbf)
            y_pred = _to_minus_plus(rbf.predict(X_test))
            metrics_accum.append(
                compute_binary_classification_metrics(y_test, y_pred, positive=1, negative=-1)
            )
        aggregated = {key: float(np.mean([m[key] for m in metrics_accum])) for key in metrics_accum[0]}
        results["rbf"][name] = aggregated

    return results
logger = logging.getLogger(__name__)
