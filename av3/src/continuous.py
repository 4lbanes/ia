from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np


Array = np.ndarray


@dataclass
class Problem:
    name: str
    func: Callable[[Array], float]
    bounds: Sequence[Tuple[float, float]]
    goal: str  # "min" ou "max"
    known_optimum: float | None = None
    rounding: int = 4

    def clamp(self, x: Array) -> Array:
        clipped = []
        for val, (low, high) in zip(x, self.bounds):
            clipped.append(np.clip(val, low, high))
        return np.asarray(clipped, dtype=float)

    def evaluate(self, x: Array) -> float:
        return float(self.func(self.clamp(x)))

    def is_better(self, cand: float, ref: float) -> bool:
        return cand < ref if self.goal == "min" else cand > ref

    def compare(self, cand: float, ref: float) -> float:
        return cand - ref if self.goal == "max" else ref - cand

    def estimate_optimum(self, rng: np.random.Generator) -> float:
        if self.known_optimum is not None:
            return self.known_optimum
        lower = np.array([b[0] for b in self.bounds], dtype=float)
        upper = np.array([b[1] for b in self.bounds], dtype=float)
        samples = rng.uniform(lower, upper, size=(5000, len(self.bounds)))
        vals = np.apply_along_axis(self.evaluate, 1, samples)
        return float(np.max(vals) if self.goal == "max" else np.min(vals))


@dataclass
class RunResult:
    best_x: Array
    best_value: float
    iterations: int


def _mode(values: Iterable[float], rounding: int) -> Tuple[float, int]:
    rounded = [round(v, rounding) for v in values]
    counts = Counter(rounded)
    val, freq = counts.most_common(1)[0]
    return float(val), freq


def _random_bounds(bounds: Sequence[Tuple[float, float]], rng: np.random.Generator) -> Array:
    lows = np.array([b[0] for b in bounds], dtype=float)
    highs = np.array([b[1] for b in bounds], dtype=float)
    return rng.uniform(lows, highs)


def _hits_optimum(value: float, problem: Problem) -> bool:
    if problem.known_optimum is None:
        return False
    tolerance = 10 ** (-problem.rounding)
    return abs(value - problem.known_optimum) <= tolerance


class HillClimbing:
    def __init__(self, epsilon: float = 0.1, max_iter: int = 1000, patience: int = 60):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.patience = patience

    def run(self, problem: Problem, rng: np.random.Generator) -> RunResult:
        current = np.array([b[0] for b in problem.bounds], dtype=float)
        best_val = problem.evaluate(current)
        no_improve = 0
        step_size = self.epsilon
        for step in range(1, self.max_iter + 1):
            # Avalia vários vizinhos na mesma iteração para aumentar a chance de atingir o ótimo.
            candidates = current + rng.uniform(-step_size, step_size, size=(64, len(current)))
            vals = np.apply_along_axis(problem.evaluate, 1, candidates)
            best_idx = int(np.argmin(vals) if problem.goal == "min" else np.argmax(vals))
            cand_val = float(vals[best_idx])
            candidate = candidates[best_idx]
            if problem.is_better(cand_val, best_val):
                current, best_val = candidate, cand_val
                no_improve = 0
            else:
                no_improve += 1
            if no_improve > 0 and no_improve % max(2, self.patience // 2) == 0:
                step_size = max(step_size * 0.6, 1e-4)
            if no_improve >= self.patience:
                return RunResult(current, best_val, step)
        return RunResult(current, best_val, self.max_iter)


class LocalRandomSearch:
    def __init__(self, sigma: float = 0.1, max_iter: int = 1000, patience: int = 60):
        self.sigma = sigma
        self.max_iter = max_iter
        self.patience = patience

    def run(self, problem: Problem, rng: np.random.Generator) -> RunResult:
        current = _random_bounds(problem.bounds, rng)
        best_val = problem.evaluate(current)
        ranges = np.array([hi - lo for lo, hi in problem.bounds], dtype=float)
        no_improve = 0
        sigma = self.sigma
        for step in range(1, self.max_iter + 1):
            candidates = current + rng.normal(0, sigma * ranges, size=(16, len(current)))
            vals = np.apply_along_axis(problem.evaluate, 1, candidates)
            best_idx = int(np.argmin(vals) if problem.goal == "min" else np.argmax(vals))
            cand_val = float(vals[best_idx])
            candidate = candidates[best_idx]
            if problem.is_better(cand_val, best_val):
                current, best_val = candidate, cand_val
                no_improve = 0
            else:
                no_improve += 1
            if no_improve > 0 and no_improve % max(2, self.patience // 2) == 0:
                sigma = max(sigma * 0.7, 1e-4)
            if no_improve >= self.patience:
                return RunResult(current, best_val, step)
        return RunResult(current, best_val, self.max_iter)


class GlobalRandomSearch:
    def __init__(self, sigma: float = 0.4, max_iter: int = 1000):
        self.sigma = sigma
        self.max_iter = max_iter

    def run(self, problem: Problem, rng: np.random.Generator) -> RunResult:
        best = _random_bounds(problem.bounds, rng)
        best_val = problem.evaluate(best)
        ranges = np.array([hi - lo for lo, hi in problem.bounds], dtype=float)
        for step in range(1, self.max_iter + 1):
            candidates = []
            # Metade das amostras puramente aleatórias, metade explorando ao redor do melhor.
            for _ in range(4):
                candidates.append(_random_bounds(problem.bounds, rng))
            scale = self.sigma * (0.995 ** step)
            for _ in range(4):
                candidates.append(best + rng.normal(0, scale * ranges))
            vals = [problem.evaluate(c) for c in candidates]
            idx = int(np.argmin(vals) if problem.goal == "min" else np.argmax(vals))
            cand_val = float(vals[idx])
            candidate = candidates[idx]
            if problem.is_better(cand_val, best_val):
                best, best_val = candidate, cand_val
        return RunResult(best, best_val, self.max_iter)


def _ackley(x: Array) -> float:
    x1, x2 = x
    term1 = -0.2 * np.sqrt(0.5 * (x1**2 + x2**2))
    term2 = 0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))
    return -20 * np.exp(term1) - np.exp(term2) + 20 + np.e


def build_problems() -> List[Problem]:
    return [
        Problem(
            name="Quadrática simples",
            func=lambda x: x[0] ** 2 + x[1] ** 2,
            bounds=[(-100.0, 100.0), (-100.0, 100.0)],
            goal="min",
            known_optimum=0.0,
            rounding=3,
        ),
        Problem(
            name="Mistura de gaussianas",
            func=lambda x: np.exp(-(x[0] ** 2 + x[1] ** 2))
            + 2 * np.exp(-((x[0] - 1.7) ** 2 + (x[1] - 1.7) ** 2)),
            bounds=[(-2.0, 4.0), (-2.0, 5.0)],
            goal="max",
            known_optimum=2.003,
            rounding=3,
        ),
        Problem(
            name="Ackley",
            func=_ackley,
            bounds=[(-8.0, 8.0), (-8.0, 8.0)],
            goal="min",
            known_optimum=0.0,
            rounding=3,
        ),
        Problem(
            name="Rastrigin 2D",
            func=lambda x: (x[0] ** 2 - 10 * np.cos(2 * np.pi * x[0]) + 10)
            + (x[1] ** 2 - 10 * np.cos(2 * np.pi * x[1]) + 10),
            bounds=[(-5.12, 5.12), (-5.12, 5.12)],
            goal="min",
            known_optimum=0.0,
            rounding=3,
        ),
        Problem(
            name="Função composta",
            func=lambda x: (x[0] ** 2 * np.cos(x[0]) / 20.0)
            + 2 * np.exp(-(x[0] ** 2) - (x[1] - 1) ** 2)
            + 0.01 * x[0] * x[1],
            bounds=[(-10.0, 10.0), (-10.0, 10.0)],
            goal="max",
            rounding=3,
        ),
        Problem(
            name="Função senoidal assimétrica",
            func=lambda x: x[0] * np.sin(4 * np.pi * x[0]) - x[1] * np.sin(4 * np.pi * x[1] + np.pi) + 1,
            bounds=[(-1.0, 3.0), (-1.0, 3.0)],
            goal="max",
            rounding=3,
        ),
    ]


def _select_hyperparameter(
    problem: Problem,
    values: Sequence[float],
    builder: Callable[[float], object],
    rng: np.random.Generator,
    warmup_runs: int = 16,
) -> float:
    results: Dict[float, float] = {}
    hits: Dict[float, int] = {}
    seed_pool = rng.integers(0, 2**32 - 1, size=warmup_runs)
    for val in values:
        bests = []
        for seed in seed_pool:
            opt = builder(val)
            res = opt.run(problem, np.random.default_rng(seed))
            bests.append(res.best_value)
        results[val] = float(np.median(bests))
        hits[val] = sum(1 for b in bests if _hits_optimum(b, problem))
    # Prefer o menor hiperparâmetro que atinge o ótimo conhecido
    feasible = [v for v, c in hits.items() if c > 0]
    if feasible:
        return float(min(feasible))
    target = max(results.values()) if problem.goal == "max" else min(results.values())
    tolerance = 1e-3 + 0.01 * abs(target)
    feasible = [v for v, res in results.items() if abs(res - target) <= tolerance]
    if feasible:
        return min(feasible)
    return float(min(results, key=lambda v: problem.compare(target, results[v])))


def run_continuous_experiments(
    runs: int = 100,
    max_iter: int = 1000,
    patience: int = 60,
    seed: int | None = None,
    output_dir: str | Path = "av3/resultados",
) -> Dict[str, Dict[str, object]]:
    rng = np.random.default_rng(seed)
    problems = build_problems()
    results: Dict[str, Dict[str, object]] = {}
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for problem in problems:
        prob_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        summaries: Dict[str, object] = {}
        hc_eps = _select_hyperparameter(
            problem,
            values=[0.5, 0.2, 0.1, 0.05, 0.01],
            builder=lambda eps: HillClimbing(eps, max_iter=max_iter, patience=patience),
            rng=prob_rng,
        )
        lrs_sigma = _select_hyperparameter(
            problem,
            values=[0.8, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001],
            builder=lambda sigma: LocalRandomSearch(sigma, max_iter=max_iter, patience=patience),
            rng=prob_rng,
        )
        grs_sigma = _select_hyperparameter(
            problem,
            values=[0.8, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001],
            builder=lambda sigma: GlobalRandomSearch(sigma, max_iter=max_iter),
            rng=prob_rng,
        )

        algo_configs = {
            "hill_climbing": HillClimbing(hc_eps, max_iter=max_iter, patience=patience),
            "local_random_search": LocalRandomSearch(lrs_sigma, max_iter=max_iter, patience=patience),
            "global_random_search": GlobalRandomSearch(grs_sigma, max_iter=max_iter),
        }

        for name, optimizer in algo_configs.items():
            best_values: List[float] = []
            best_points: List[List[float]] = []
            for run_idx in range(runs):
                run_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
                res = optimizer.run(problem, run_rng)
                best_values.append(res.best_value)
                best_points.append(res.best_x.tolist())

            mode_val, freq = _mode(best_values, problem.rounding)
            success_count = sum(1 for v in best_values if _hits_optimum(v, problem))
            summaries[name] = {
                "hyperparameter": getattr(optimizer, "epsilon", None) or getattr(optimizer, "sigma", None),
                "mode": mode_val,
                "mode_frequency": freq,
                "best_value": float(np.max(best_values) if problem.goal == "max" else np.min(best_values)),
                "mean_value": float(np.mean(best_values)),
                "std_value": float(np.std(best_values)),
                "all_best_values": best_values,
                "best_points": best_points,
                "runs": runs,
                "max_iter": max_iter,
                "patience": patience,
                "goal": problem.goal,
                "known_optimum": problem.known_optimum,
                "success_tolerance": 10 ** (-problem.rounding),
                "success_count": success_count,
                "success_rate": success_count / runs,
            }

        results[problem.name] = summaries

    out_path = Path(output_dir) / "continuous_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


if __name__ == "__main__":
    summary = run_continuous_experiments()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
