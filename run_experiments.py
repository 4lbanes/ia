from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

from av2.experiments import (
    Stage1Config,
    Stage2Config,
    run_full_stage1,
    run_full_stage2,
    stage1_results_to_tables,
    stage2_results_to_table,
)
from av2.stage1 import Stage1Results
from av2.stage2 import Stage2Results
from av2.visualization import (
    plot_confusion_heatmap,
    plot_learning_curve,
    plot_metric_boxplot,
    plot_scatter_2d,
)


logger = logging.getLogger(__name__)


def _save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info("Arquivo salvo em %s", path)


def _stage1_plots(results: Stage1Results, output_dir: Path) -> None:
    scatter_path = output_dir / "stage1" / "scatter.png"
    scatter_path.parent.mkdir(parents=True, exist_ok=True)
    plot_scatter_2d(results.dataset.features, results.dataset.labels, path=scatter_path)

    for model_name, model_result in results.model_results.items():
        model_dir = output_dir / "stage1" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Boxplots for metrics
        metrics_data = {metric: values.tolist() for metric, values in model_result.metrics.items()}
        plot_metric_boxplot(
            metrics_data,
            path=model_dir / "metric_boxplot.png",
            title=f"Distribuição de Métricas ({model_name})",
            ylabel="Valor",
        )

        for metric in model_result.metrics.keys():
            confusion_best = model_result.confusion_for_metric(metric, best=True)
            confusion_worst = model_result.confusion_for_metric(metric, best=False)

            plot_confusion_heatmap(
                confusion_best,
                labels=["-1", "1"],
                path=model_dir / f"confusion_best_{metric}.png",
                title=f"Matriz de Confusão - Melhor {metric} ({model_name})",
            )
            plot_confusion_heatmap(
                confusion_worst,
                labels=["-1", "1"],
                path=model_dir / f"confusion_worst_{metric}.png",
                title=f"Matriz de Confusão - Pior {metric} ({model_name})",
            )

            history_best = model_result.learning_curve(metric, best=True)
            history_worst = model_result.learning_curve(metric, best=False)

            if "train_accuracy" in history_best:
                plot_learning_curve(
                    history_best.get("train_accuracy", []),
                    history_best.get("val_accuracy", []),
                    path=model_dir / f"learning_curve_best_{metric}.png",
                    title=f"Curva de Aprendizagem (Melhor {metric}) - {model_name}",
                )
            if "train_accuracy" in history_worst:
                plot_learning_curve(
                    history_worst.get("train_accuracy", []),
                    history_worst.get("val_accuracy", []),
                    path=model_dir / f"learning_curve_worst_{metric}.png",
                    title=f"Curva de Aprendizagem (Pior {metric}) - {model_name}",
                )


def _stage2_plots(results: Stage2Results, output_dir: Path) -> None:
    num_classes = results.model_results["mlp"].records[0].confusion.shape[0]
    labels = [str(idx) for idx in range(num_classes)]

    for model_name, model_result in results.model_results.items():
        model_dir = output_dir / "stage2" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        plot_metric_boxplot(
            {model_name: model_result.accuracies.tolist()},
            path=model_dir / "accuracy_boxplot.png",
            title=f"Acurácia por rodada - {model_name.upper()}",
            ylabel="Acurácia",
        )

        plot_confusion_heatmap(
            model_result.confusion(best=True),
            labels=labels,
            path=model_dir / "confusion_best.png",
            title=f"Matriz de Confusão - Melhor Acurácia ({model_name})",
            annot=False,
        )
        plot_confusion_heatmap(
            model_result.confusion(best=False),
            labels=labels,
            path=model_dir / "confusion_worst.png",
            title=f"Matriz de Confusão - Pior Acurácia ({model_name})",
            annot=False,
        )

        history_best = model_result.learning_curve(best=True)
        history_worst = model_result.learning_curve(best=False)

        if "train_accuracy" in history_best:
            plot_learning_curve(
                history_best.get("train_accuracy", []),
                history_best.get("val_accuracy", []),
                path=model_dir / "learning_curve_best.png",
                title=f"Curva de Aprendizagem (Melhor Acurácia) - {model_name.upper()}",
            )
        if "train_accuracy" in history_worst:
            plot_learning_curve(
                history_worst.get("train_accuracy", []),
                history_worst.get("val_accuracy", []),
                path=model_dir / "learning_curve_worst.png",
                title=f"Curva de Aprendizagem (Pior Acurácia) - {model_name.upper()}",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa os experimentos das etapas AV2.")
    parser.add_argument("--stage1-dataset", type=Path, help="Caminho para o arquivo spiral_d.csv.")
    parser.add_argument("--stage2-dataset", type=Path, help="Diretório raiz contendo as imagens RecFac.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Diretório para salvar métricas e gráficos.")
    parser.add_argument("--stage1-runs", type=int, default=500, help="Número de rodadas Monte Carlo para a Etapa 1.")
    parser.add_argument("--stage2-runs", type=int, default=10, help="Número de rodadas Monte Carlo para a Etapa 2.")
    parser.add_argument("--image-size", type=int, nargs=2, default=(40, 40), help="Dimensões de redimensionamento das imagens (linhas colunas).")
    parser.add_argument("--skip-plots", action="store_true", help="Apenas gerar métricas em tabelas, sem gráficos.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Nível de log exibido no terminal.",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Output directory configurado para %s", args.output_dir)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage1_dataset:
        logger.info(
            "Rodando Stage 1 com dataset=%s e %d rodadas",
            args.stage1_dataset,
            args.stage1_runs,
        )
        stage1_config = Stage1Config(
            dataset_path=str(args.stage1_dataset),
            monte_carlo_runs=args.stage1_runs,
        )
        stage1_results = run_full_stage1(stage1_config)
        tables = stage1_results_to_tables(stage1_results)
        _save_json(tables, output_dir / "stage1" / "metric_tables.json")

        if not args.skip_plots:
            logger.info("Gerando gráficos da Stage 1")
            _stage1_plots(stage1_results, output_dir)
        logger.info("Stage 1 concluída com sucesso")

    if args.stage2_dataset:
        logger.info(
            "Rodando Stage 2 com dataset=%s, %d rodadas e image_size=%s",
            args.stage2_dataset,
            args.stage2_runs,
            tuple(args.image_size),
        )
        stage2_config = Stage2Config(
            dataset_dir=str(args.stage2_dataset),
            monte_carlo_runs=args.stage2_runs,
            image_size=tuple(args.image_size),
        )
        stage2_results = run_full_stage2(stage2_config)
        table = stage2_results_to_table(stage2_results)
        _save_json(table, output_dir / "stage2" / "metric_tables.json")

        if not args.skip_plots:
            logger.info("Gerando gráficos da Stage 2")
            _stage2_plots(stage2_results, output_dir)
        logger.info("Stage 2 concluída com sucesso")


if __name__ == "__main__":
    main()
