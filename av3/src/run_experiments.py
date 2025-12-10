from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

try:
    # Execução como módulo: python -m av3.run_experiments
    from .continuous import run_continuous_experiments
    from .genetic_tsp import DEFAULT_CSV_PATH, solve_and_save as solve_tsp
    from .simulated_annealing import solve_and_save as solve_sa
except ImportError:
    # Execução direta: python av3/run_experiments.py
    import sys
    import pathlib

    sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
    from av3.src.continuous import run_continuous_experiments
    from av3.src.genetic_tsp import DEFAULT_CSV_PATH, solve_and_save as solve_tsp
    from av3.src.simulated_annealing import solve_and_save as solve_sa


def parse_sections(raw: List[str] | None) -> List[str]:
    if not raw:
        return ["continuous", "sa", "ga"]
    if "all" in raw:
        return ["continuous", "sa", "ga"]
    return raw


def _safe_json(data: object) -> str:
    def _default(obj):  # type: ignore[return-type]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    return json.dumps(data, indent=2, ensure_ascii=False, default=_default)


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa os experimentos da AV3 com parâmetros padrão do trabalho.")
    parser.add_argument("--sections", nargs="+", choices=["continuous", "sa", "ga", "all"], help="Quais partes executar.")
    args = parser.parse_args()

    sections = parse_sections(args.sections)
    output_root = Path("av3/resultados")
    output_root.mkdir(parents=True, exist_ok=True)
    seed = None  # uso sempre sem semente fixa, aderindo à aleatoriedade esperada

    if "continuous" in sections:
        cont_results = run_continuous_experiments(
            runs=100,  # conforme especificação: 100 rodadas por algoritmo
            max_iter=1000,
            patience=60,
            seed=seed,
            output_dir=output_root,
        )
        print("Resultados de funções contínuas salvos em", output_root / "continuous_results.json")
        print(_safe_json(cont_results))

    if "sa" in sections:
        sa_results = solve_sa(
            output_dir=output_root,
            seed=seed,
            temp0=5.0,
            cooling=0.995,
            max_iter=5000,
            find_all=True,
            find_all_max_runs=2000,
        )
        print("Resultados da Têmpera Simulada salvos em", output_root / "simulated_annealing.json")
        print(_safe_json(sa_results))

    if "ga" in sections:
        tsp_results = solve_tsp(
            csv_path=str(DEFAULT_CSV_PATH),
            output_dir=output_root,
            seed=seed,
            points_per_region=34,  # 30 < Npontos < 60, escolhemos 34
            acceptable_epsilon=0.05,
            target_length=None,
            runs=10,
        )
        print("Resultados do GA (TSP 3D) salvos em", output_root / "genetic_tsp.json")
        print(_safe_json(tsp_results))


if __name__ == "__main__":
    main()
