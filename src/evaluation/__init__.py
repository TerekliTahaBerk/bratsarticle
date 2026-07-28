"""Central, patient-level BraTS segmentation evaluation."""

from evaluation.config import EvaluationConfig, load_evaluation_config
from evaluation.evaluator import CentralEvaluator, summarize_patient_metrics

__all__ = [
    "CentralEvaluator",
    "EvaluationConfig",
    "load_evaluation_config",
    "summarize_patient_metrics",
]
