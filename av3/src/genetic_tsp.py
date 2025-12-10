from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


PointCloud = np.ndarray
Route = List[int]
DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "CaixeiroGruposGA.csv"


def _to_serializable(obj):  # type: ignore[return-type]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


@dataclass
class GAConfig:
    population_size: int = 120
    generations: int = 400
    tournament_size: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.01
    elitism: int = 2
    acceptable_epsilon: float = 0.05  # tolerância relativa (ε) usada no critério de parada aceitável


def _load_csv_points(csv_path: Path, points_per_region: int) -> PointCloud | None:
    if csv_path.exists():
        data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
        if data.dtype.names:
            data = np.column_stack([data[name] for name in data.dtype.names])
        if data.ndim == 2 and data.shape[1] >= 3:
            # Se o CSV tiver uma 4ª coluna indicando a região, amostramos até points_per_region por região (>0).
            if data.shape[1] >= 4:
                regions = data[:, 3].astype(int)
                selected = []
                for region in sorted(r for r in np.unique(regions) if r > 0)[:4]:
                    idx = np.where(regions == region)[0][:points_per_region]
                    if idx.size:
                        selected.append(data[idx, :3])
                if selected:
                    return np.vstack(selected)
            # Caso contrário, pega os primeiros points_per_region*4 pontos como fallback.
            return data[: points_per_region * 4, :3]
    return None


def load_points(path: str | Path | None, points_per_region: int, rng: np.random.Generator) -> PointCloud:
    candidates = []
    if path:
        candidates.append(Path(path))
    else:
        candidates.append(DEFAULT_CSV_PATH)
        candidates.append(DEFAULT_CSV_PATH.with_name("CaixeiroGrupos.csv"))

    for csv_path in candidates:
        data = _load_csv_points(csv_path, points_per_region)
        if data is not None:
            return data

    raise FileNotFoundError("Arquivo de pontos obrigatório não encontrado (esperado CaixeiroGrupos*.csv).")


def route_length(route: Sequence[int], points: PointCloud) -> float:
    origin = np.zeros(3)
    total = float(np.linalg.norm(points[route[0]] - origin))
    for a, b in zip(route[:-1], route[1:]):
        total += float(np.linalg.norm(points[b] - points[a]))
    total += float(np.linalg.norm(points[route[-1]] - origin))
    return total


def estimate_target_length(points: PointCloud, rng: np.random.Generator, samples: int = 512) -> float:
    num_points = len(points)
    best = float("inf")
    for _ in range(samples):
        route = list(rng.permutation(num_points))
        length = route_length(route, points)
        if length < best:
            best = length
    return best


def tournament(population: List[Route], lengths: List[float], size: int, rng: np.random.Generator) -> Route:
    contenders = rng.choice(len(population), size=size, replace=False)
    best_idx = min(contenders, key=lambda idx: lengths[idx])
    return population[best_idx]


def crossover(parent1: Route, parent2: Route, rng: np.random.Generator) -> Route:
    n = len(parent1)
    cut1, cut2 = sorted(rng.choice(n, size=2, replace=False))
    child = [-1] * n
    child[cut1:cut2] = parent1[cut1:cut2]
    fill_positions = [i for i in range(n) if child[i] == -1]
    fill_values = [gene for gene in parent2 if gene not in child]
    for pos, gene in zip(fill_positions, fill_values):
        child[pos] = gene
    return child


def mutate(route: Route, rng: np.random.Generator, rate: float) -> None:
    if rng.random() < rate:
        i, j = rng.choice(len(route), size=2, replace=False)
        route[i], route[j] = route[j], route[i]


def run_ga(
    points: PointCloud,
    config: GAConfig,
    seed: int | None = None,
    target_length: float | None = None,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    num_points = len(points)
    population: List[Route] = [list(rng.permutation(num_points)) for _ in range(config.population_size)]
    lengths = [route_length(route, points) for route in population]

    best_idx = int(np.argmin(lengths))
    best_length = lengths[best_idx]
    best_route = population[best_idx].copy()
    if target_length is None:
        target_length = estimate_target_length(points, rng)
    epsilon_abs = abs(target_length) * config.acceptable_epsilon
    acceptable_threshold = target_length + epsilon_abs  # |f(x*) - ε| para minimização

    acceptable_generation = 0 if best_length <= acceptable_threshold else None
    history: List[float] = [best_length]

    if acceptable_generation is not None:
        return {
            "best_route": [int(g) for g in best_route],
            "best_length": float(best_length),
            "history": [float(v) for v in history],
            "acceptable_generation": int(acceptable_generation),
            "target_length": float(target_length),
            "acceptable_threshold": float(acceptable_threshold),
        }

    for gen in range(1, config.generations + 1):
        ranked = sorted(zip(population, lengths), key=lambda t: t[1])
        elites = [route.copy() for route, _ in ranked[: config.elitism]]

        new_pop: List[Route] = elites
        while len(new_pop) < config.population_size:
            parent1 = tournament(population, lengths, config.tournament_size, rng)
            parent2 = tournament(population, lengths, config.tournament_size, rng)
            child = parent1.copy()
            if rng.random() < config.crossover_rate:
                child = crossover(parent1, parent2, rng)
            mutate(child, rng, config.mutation_rate)
            new_pop.append(child)

        population = new_pop
        lengths = [route_length(route, points) for route in population]
        best_idx = int(np.argmin(lengths))
        if lengths[best_idx] < best_length:
            best_length = lengths[best_idx]
            best_route = population[best_idx].copy()
        history.append(best_length)
        if acceptable_generation is None and best_length <= acceptable_threshold:
            acceptable_generation = int(gen)
            break

    return {
        "best_route": [int(g) for g in best_route],
        "best_length": float(best_length),
        "history": [float(v) for v in history],
        "acceptable_generation": acceptable_generation if acceptable_generation is None else int(acceptable_generation),
        "target_length": float(target_length),
        "acceptable_threshold": float(acceptable_threshold),
    }


def multiple_runs(
    points: PointCloud,
    config: GAConfig,
    runs: int = 10,
    seed: int | None = None,
    target_length: float | None = None,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    reference_target = target_length if target_length is not None else estimate_target_length(points, rng)
    acceptable_threshold = reference_target + abs(reference_target) * config.acceptable_epsilon
    best_overall = None
    best_route: Route | None = None
    generations: List[int] = []
    histories: List[List[float]] = []

    for _ in range(runs):
        res = run_ga(
            points,
            config,
            seed=int(rng.integers(0, 2**32 - 1)),
            target_length=reference_target,
        )
        histories.append(res["history"])  # type: ignore[index]
        if res["acceptable_generation"] is not None:
            generations.append(int(res["acceptable_generation"]))
        if best_overall is None or res["best_length"] < best_overall:
            best_overall = float(res["best_length"])
            best_route = list(res["best_route"])  # type: ignore[list-item]

    mode_gen, freq = (None, 0)
    if generations:
        counts = Counter(generations)
        mode_gen, freq = counts.most_common(1)[0]

    return {
        "best_length": best_overall,
        "best_route": best_route,
        "mode_generation": mode_gen,
        "mode_frequency": freq,
        "generations_hit": generations,
        "histories": histories,
        "target_length": float(reference_target),
        "acceptable_threshold": float(acceptable_threshold),
        "runs": runs,
    }


def solve_and_save(
    csv_path: str | Path | None = DEFAULT_CSV_PATH,
    output_dir: str | Path = "av3/resultados",
    seed: int | None = None,
    points_per_region: int = 34,
    acceptable_epsilon: float = 0.05,
    target_length: float | None = None,
    runs: int = 10,
) -> Dict[str, object]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    points = load_points(csv_path, points_per_region, rng)
    config = GAConfig(acceptable_epsilon=acceptable_epsilon)
    summary = multiple_runs(points, config=config, runs=runs, seed=seed, target_length=target_length)
    payload = {
        "config": {
            **config.__dict__,
            "runs": runs,
            "points_per_region": points_per_region,
            "num_points": len(points),
        },
        "points": points.tolist(),
        "summary": summary,
    }
    out_path = Path(output_dir) / "genetic_tsp.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_to_serializable)
    return payload


if __name__ == "__main__":
    data = solve_and_save()
    print(json.dumps(data, indent=2, ensure_ascii=False, default=_to_serializable))
