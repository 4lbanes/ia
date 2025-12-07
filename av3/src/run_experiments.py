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
    parser = argparse.ArgumentParser(description="Executa os experimentos da AV3.")
    parser.add_argument("--sections", nargs="+", choices=["continuous", "sa", "ga", "all"], help="Quais partes executar.")
    parser.add_argument("--runs", type=int, default=100, help="Rodadas por algoritmo nas funções contínuas.")
    parser.add_argument("--max-iter", type=int, default=1000, help="Máximo de iterações dos algoritmos contínuos.")
    parser.add_argument("--patience", type=int, default=60, help="Parada antecipada sem melhoria (contínuos).")
    parser.add_argument("--seed", type=int, default=None, help="Semente global opcional.")
    parser.add_argument("--output-dir", default="av3/resultados", help="Diretório para salvar os resultados.")
    parser.add_argument("--sa-temp0", type=float, default=5.0, help="Temperatura inicial (SA 8-rainhas).")
    parser.add_argument("--sa-cooling", type=float, default=0.995, help="Fator de resfriamento (SA 8-rainhas).")
    parser.add_argument("--sa-max-iter", type=int, default=5000, help="Máximo de iterações (SA 8-rainhas).")
    parser.add_argument(
        "--sa-max-runs",
        type=int,
        default=2000,
        help="Máximo de execuções para buscar as 92 soluções (SA 8-rainhas).",
    )
    parser.add_argument("--sa-find-all", dest="sa_find_all", action="store_true", help="Busca pelas 92 soluções com SA.")
    parser.add_argument(
        "--sa-skip-find-all",
        dest="sa_find_all",
        action="store_false",
        help="Pula a busca das 92 soluções (não recomendável, apenas para execuções rápidas).",
    )
    parser.set_defaults(sa_find_all=True)
    parser.add_argument(
        "--tsp-csv",
        default=str(DEFAULT_CSV_PATH),
        help="Caminho para CaixeiroGruposGA.csv (padrão: arquivo em av3/data).",
    )
    parser.add_argument(
        "--tsp-acceptable-epsilon",
        type=float,
        default=0.05,
        help="Tolerância relativa ε para aceitar solução (slide: |f(x*) - ε|). Se alvo é desconhecido, exige melhora de (1-ε) sobre a melhor rota inicial.",
    )
    parser.add_argument(
        "--tsp-target-length",
        type=float,
        default=None,
        help="Valor de referência para o custo ótimo (se conhecido). Sem isso, usa a melhor rota inicial como referência.",
    )
    parser.add_argument("--tsp-points-per-region", type=int, default=40, help="Quantidade de pontos por região gerada.")
    args = parser.parse_args()

    sections = parse_sections(args.sections)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if "continuous" in sections:
        cont_results = run_continuous_experiments(
            runs=args.runs,
            max_iter=args.max_iter,
            patience=args.patience,
            seed=args.seed,
            output_dir=output_root,
        )
        print("Resultados de funções contínuas salvos em", output_root / "continuous_results.json")
        print(_safe_json(cont_results))

    if "sa" in sections:
        sa_results = solve_sa(
            output_dir=output_root,
            seed=args.seed,
            temp0=args.sa_temp0,
            cooling=args.sa_cooling,
            max_iter=args.sa_max_iter,
            find_all=args.sa_find_all,
            find_all_max_runs=args.sa_max_runs,
        )
        print("Resultados da Têmpera Simulada salvos em", output_root / "simulated_annealing.json")
        print(_safe_json(sa_results))

    if "ga" in sections:
        tsp_results = solve_tsp(
            csv_path=args.tsp_csv,
            output_dir=output_root,
            seed=args.seed,
            points_per_region=args.tsp_points_per_region,
            acceptable_epsilon=args.tsp_acceptable_epsilon,
            target_length=args.tsp_target_length,
        )
        print("Resultados do GA (TSP 3D) salvos em", output_root / "genetic_tsp.json")
        print(_safe_json(tsp_results))


if __name__ == "__main__":
    main()
