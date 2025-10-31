from __future__ import annotations

"""High-level helpers to execute Stage 1 and Stage 2 experiments."""

from typing import Dict

from .metrics import MetricSummary
from .stage1 import Stage1Config, Stage1Results, run_stage1_monte_carlo
from .stage2 import Stage2Config, Stage2Results, run_stage2_monte_carlo


def _metric_summary_to_dict(summary: MetricSummary) -> Dict[str, float]:
    return {
        "mean": summary.mean,
        "std": summary.std,
        "maximum": summary.maximum,
        "minimum": summary.minimum,
    }


def run_full_stage1(config: Stage1Config) -> Stage1Results:
    """Convenience wrapper to run Stage 1 Monte Carlo evaluation."""
    return run_stage1_monte_carlo(config)


def run_full_stage2(config: Stage2Config) -> Stage2Results:
    """Convenience wrapper to run Stage 2 Monte Carlo evaluation."""
    return run_stage2_monte_carlo(config)


def stage1_results_to_tables(results: Stage1Results) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Convert Stage 1 results into table-ready dictionaries."""
    tables: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model_name, model_result in results.model_results.items():
        tables[model_name] = {}
        for metric, summary in model_result.metric_summaries.items():
            tables[model_name][metric] = _metric_summary_to_dict(summary)
    return tables


def stage2_results_to_table(results: Stage2Results) -> Dict[str, Dict[str, float]]:
    """Convert Stage 2 accuracy summaries into table-ready dictionaries."""
    table: Dict[str, Dict[str, float]] = {}
    for model_name, model_result in results.model_results.items():
        table[model_name] = _metric_summary_to_dict(model_result.summary)
    return table
