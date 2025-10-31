from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import trange

from .data import load_recfac_dataset, one_hot_encode, train_test_split
from .metrics import MetricSummary, confusion_matrix, summarise_metric
from .models.adaline import Adaline, AdalineConfig
from .models.mlp import MLP, MLPConfig
from .models.rbf import RBFNetwork, RBFConfig


logger = logging.getLogger(__name__)


def _aggregate_curves(curves: List[List[float]]) -> List[float]:
    if not curves:
        return []
    max_len = max(len(curve) for curve in curves)
    aggregated: List[float] = []
    for idx in range(max_len):
        values = []
        for curve in curves:
            if idx < len(curve):
                values.append(curve[idx])
            else:
                values.append(curve[-1])
        aggregated.append(float(np.mean(values)))
    return aggregated


class OneVsRestAdaline:
    """One-vs-rest multi-class classifier using multiple ADALINE models."""

    def __init__(self, n_classes: int, config: AdalineConfig) -> None:
        self.n_classes = n_classes
        self.base_config = config
        self.models: List[Adaline] = []
        self.training_history: Dict[str, List[float]] = {
            "train_accuracy": [],
            "val_accuracy": [],
            "mse_per_epoch": [],
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        *,
        seed_offset: int = 0,
    ) -> None:
        self.models = []
        train_curves: List[List[float]] = []
        val_curves: List[List[float]] = []
        mse_curves: List[List[float]] = []

        for class_idx in range(self.n_classes):
            binary_train = np.where(y_train == class_idx, 1, -1)
            binary_val = None
            if y_val is not None:
                binary_val = np.where(y_val == class_idx, 1, -1)

            config = replace(
                self.base_config,
                random_seed=self.base_config.random_seed + class_idx + seed_offset,
            )
            model = Adaline(X_train, binary_train, config)
            model.fit(X_val=X_val, y_val=binary_val if binary_val is not None else None)
            self.models.append(model)

            train_curves.append(model.training_history.get("train_accuracy", []))
            val_curves.append(model.training_history.get("val_accuracy", []))
            mse_curves.append(model.training_history.get("mse_per_epoch", []))

        self.training_history = {
            "train_accuracy": _aggregate_curves(train_curves),
            "val_accuracy": _aggregate_curves(val_curves),
            "mse_per_epoch": _aggregate_curves(mse_curves),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = []
        for model in self.models:
            score = model.decision_function(X)
            scores.append(score)
        stacked = np.vstack(scores)
        return np.argmax(stacked, axis=0)


@dataclass
class Stage2Config:
    dataset_dir: str
    image_size: Tuple[int, int] = (40, 40)
    monte_carlo_runs: int = 10
    test_size: float = 0.2
    random_seed: int = 42
    normalize: bool = True
    adaline_config: AdalineConfig = field(
        default_factory=lambda: AdalineConfig(learning_rate=5e-3, max_epochs=800, plot_training=False)
    )
    mlp_config: MLPConfig = field(
        default_factory=lambda: MLPConfig(hidden_layers=[128, 64], activation="sigmoid", max_epochs=200)
    )
    rbf_config: RBFConfig = field(
        default_factory=lambda: RBFConfig(num_centers=80, max_epochs=200, learning_rate=5e-3)
    )


@dataclass
class Stage2ModelRunRecord:
    y_true: np.ndarray
    y_pred: np.ndarray
    confusion: np.ndarray
    history: Dict[str, List[float]]


@dataclass
class Stage2ModelResult:
    accuracies: np.ndarray
    records: List[Stage2ModelRunRecord]
    summary: MetricSummary
    best_index: int
    worst_index: int

    def confusion(self, best: bool = True) -> np.ndarray:
        idx = self.best_index if best else self.worst_index
        return self.records[idx].confusion

    def learning_curve(self, best: bool = True) -> Dict[str, List[float]]:
        idx = self.best_index if best else self.worst_index
        return self.records[idx].history


@dataclass
class Stage2Results:
    config: Stage2Config
    dataset_metadata: Dict[str, object]
    model_results: Dict[str, Stage2ModelResult]


def run_stage2_monte_carlo(config: Stage2Config) -> Stage2Results:
    logger.info(
        "Carregando dataset RecFac em %s (image_size=%s, normalize=%s)",
        config.dataset_dir,
        config.image_size,
        config.normalize,
    )
    dataset = load_recfac_dataset(
        config.dataset_dir,
        image_size=config.image_size,
        normalize=config.normalize,
        flatten=True,
    )
    X = dataset.features
    y = dataset.labels
    num_classes = len(np.unique(y))

    rng = np.random.default_rng(config.random_seed)
    runs = config.monte_carlo_runs
    logger.info("Iniciando simulação Stage 2 com %d rodadas (seed base=%d)", runs, config.random_seed)

    accuracies: Dict[str, np.ndarray] = {
        "adaline": np.zeros(runs),
        "mlp": np.zeros(runs),
        "rbf": np.zeros(runs),
    }
    records: Dict[str, List[Stage2ModelRunRecord]] = {name: [] for name in accuracies}

    labels = sorted(np.unique(y))

    for run in trange(runs, desc="Etapa 2 - Monte Carlo", leave=False):
        seed = int(rng.integers(0, 1_000_000_000))
        logger.info("Stage 2 - Rodada %d/%d (seed=%d)", run + 1, runs, seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_seed=seed
        )

        # ADALINE (one-vs-rest)
        adaline_model = OneVsRestAdaline(num_classes, replace(config.adaline_config, random_seed=seed))
        adaline_model.fit(X_train, y_train, X_val=X_test, y_val=y_test, seed_offset=run)
        y_pred_ada = adaline_model.predict(X_test)
        acc_ada = float(np.mean(y_pred_ada == y_test))
        accuracies["adaline"][run] = acc_ada
        records["adaline"].append(
            Stage2ModelRunRecord(
                y_true=y_test,
                y_pred=y_pred_ada,
                confusion=confusion_matrix(y_test, y_pred_ada, labels=labels),
                history=adaline_model.training_history,
            )
        )

        # MLP
        mlp_cfg = replace(config.mlp_config, random_seed=seed)
        mlp = MLP(X_train.shape[0], num_classes, mlp_cfg)
        y_train_mlp = one_hot_encode(y_train, num_classes)
        y_test_mlp = one_hot_encode(y_test, num_classes)
        mlp.fit(X_train, y_train_mlp, X_val=X_test, y_val=y_test_mlp)
        y_pred_mlp = mlp.predict(X_test)
        acc_mlp = float(np.mean(y_pred_mlp == y_test))
        accuracies["mlp"][run] = acc_mlp
        records["mlp"].append(
            Stage2ModelRunRecord(
                y_true=y_test,
                y_pred=y_pred_mlp,
                confusion=confusion_matrix(y_test, y_pred_mlp, labels=labels),
                history=mlp.training_history,
            )
        )

        # RBF
        rbf_cfg = replace(config.rbf_config, random_seed=seed)
        rbf = RBFNetwork(config=rbf_cfg)
        rbf.fit(X_train, y_train_mlp, X_val=X_test, y_val=y_test_mlp)
        y_pred_rbf = rbf.predict(X_test)
        acc_rbf = float(np.mean(y_pred_rbf == y_test))
        accuracies["rbf"][run] = acc_rbf
        records["rbf"].append(
            Stage2ModelRunRecord(
                y_true=y_test,
                y_pred=y_pred_rbf,
                confusion=confusion_matrix(y_test, y_pred_rbf, labels=labels),
                history=rbf.training_history,
            )
        )

    model_results: Dict[str, Stage2ModelResult] = {}
    for name, acc_values in accuracies.items():
        best_index = int(np.argmax(acc_values))
        worst_index = int(np.argmin(acc_values))
        summary = summarise_metric(acc_values)
        model_results[name] = Stage2ModelResult(
            accuracies=acc_values,
            records=records[name],
            summary=summary,
            best_index=best_index,
            worst_index=worst_index,
        )

    logger.info("Concluídas as %d rodadas da Stage 2", runs)

    return Stage2Results(
        config=config,
        dataset_metadata=dataset.metadata,
        model_results=model_results,
    )
