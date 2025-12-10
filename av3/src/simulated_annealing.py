from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

BOARD_SIZE = 8
MAX_PAIRS = 28


def attacking_pairs(board: Iterable[int]) -> int:
    queens = list(board)
    conflicts = 0
    for c1 in range(BOARD_SIZE):
        for c2 in range(c1 + 1, BOARD_SIZE):
            r1, r2 = queens[c1], queens[c2]
            if r1 == r2 or abs(r1 - r2) == abs(c1 - c2):
                conflicts += 1
    return conflicts


def fitness(board: Iterable[int]) -> float:
    return float(MAX_PAIRS - attacking_pairs(board))


@dataclass
class SARunResult:
    board: List[int]
    score: float
    iterations: int
    history: List[float]


class SimulatedAnnealing:
    def __init__(self, temp0: float = 5.0, cooling: float = 0.995, max_iter: int = 5000):
        self.temp0 = temp0
        self.cooling = cooling
        self.max_iter = max_iter

    @staticmethod
    def _neighbor(board: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        candidate = board.copy()
        c1, c2 = rng.choice(BOARD_SIZE, size=2, replace=False)
        candidate[c1], candidate[c2] = candidate[c2], candidate[c1]
        return candidate

    def run(self, seed: int | None = None, target: float = MAX_PAIRS) -> SARunResult:
        rng = np.random.default_rng(seed)
        board = rng.permutation(BOARD_SIZE)
        current_score = fitness(board)
        best_board = board.copy()
        best_score = current_score
        temperature = self.temp0
        history: List[float] = [current_score]

        for iteration in range(1, self.max_iter + 1):
            candidate = self._neighbor(board, rng)
            cand_score = fitness(candidate)
            delta = cand_score - current_score
            accept = delta > 0
            if not accept:
                if temperature > 1e-9:
                    accept = rng.random() < np.exp(delta / max(temperature, 1e-9))
            if accept:
                board = candidate
                current_score = cand_score
                if cand_score > best_score:
                    best_board = candidate
                    best_score = cand_score
            temperature *= self.cooling
            history.append(best_score)
            if best_score >= target:
                break

        return SARunResult(best_board.tolist(), best_score, iteration, history)


def find_all_solutions(
    solver: SimulatedAnnealing,
    seed: int | None = None,
    max_runs: int | None = None,
    target_count: int = 92,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    solutions: Dict[Tuple[int, ...], int] = {}
    total_iterations = 0
    runs_done = 0
    while True:
        res = solver.run(seed=int(rng.integers(0, 2**32 - 1)))
        total_iterations += res.iterations
        runs_done += 1
        if res.score >= MAX_PAIRS:
            key = tuple(res.board)
            solutions.setdefault(key, 0)
            solutions[key] += 1
        if len(solutions) >= target_count:
            break
        if max_runs is not None and runs_done >= max_runs:
            break
    return {
        "found": len(solutions),
        "solutions": [list(sol) for sol in solutions],
        "visits_per_solution": {",".join(map(str, sol)): count for sol, count in solutions.items()},
        "runs_performed": runs_done,
        "avg_iterations": total_iterations / max(1, runs_done),
    }


def solve_and_save(
    output_dir: str | Path = "av3/resultados",
    seed: int | None = None,
    temp0: float = 10.0,
    cooling: float = 0.997,
    max_iter: int = 20000,
    find_all: bool = True,
    find_all_max_runs: int | None = None,
) -> Dict[str, object]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    solver = SimulatedAnnealing(temp0=temp0, cooling=cooling, max_iter=max_iter)
    best = solver.run(seed=seed)
    all_found = find_all_solutions(solver, seed=seed, max_runs=find_all_max_runs) if find_all else None
    payload = {
        "single_run": {
            "board": best.board,
            "score": best.score,
            "iterations": best.iterations,
        },
        "all_solutions": all_found,
        "config": {"temp0": temp0, "cooling": cooling, "max_iter": max_iter},
    }
    out_path = Path(output_dir) / "simulated_annealing.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


if __name__ == "__main__":
    data = solve_and_save()
    print(json.dumps(data, indent=2, ensure_ascii=False))
