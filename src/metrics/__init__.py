"""Custom metrics for evaluating anomaly detection models.

This module provides various metrics for evaluating anoaly detection performance.

- Area Under Curve (AUC) metric:
    - ``AUROC``: Area Under Reciever Operating Characteristics curve
    - ``AUPR``: Area Under Precision-Recall curve
    - ``AUPRO``: Area Under Pre-Region Overlap Curve
    - ``AUPIMO``: Area Under Per-Image Missed Overlap curve

- F1-score metrics:
    - ``F1Score``: Standard F1 score
    - ``F1Max``: Maximum F1 score across thresholds

- Threshold metrics:
    - ``F1AdaptiveThreshold``: Finds optimal threshold by maximizing F1 score
    - ``ManualThreshold``: Uses manually specified thresholds

- Other metrics: 
    - ``AnomalibMetric``: Base class for custom metrics
    - ``AnomalyScoreDistribution``: Analyzes score distributions
    - ``BinaryPrecisionRecallCurve``: Computes precission-recall curves
    - ``Elevator``: Combines multile metrics for evaluation
    - ``MinMax``: Normalizes scores to [0, 1] range
    - ``PRO``: Per-Region Overlap score
    - ``PIMO``: Per-Image Overlap score

Example: 
    >>> from src.metrics import AUROC, F1Score
    >>> auroc = AUROC()
    >>> f1 = F1SCORE()
    >>> lables = torch.Tensor([0, 1, 0, 1])
    >>> scores = torch.Tensor([0.1, 0.9, 0.2, 0.8])
    >>> auroc(scores, labels)
    tensor(1.)
    >>> f1(scores, labels)
    tensor(1.)
"""

from .min_max import MinMax
from .threshold import F1AdaptiveThreshold

__all__ = [
    # "AUROC",
    # "AUPR",
    # "AUPRO",
    # "AnomalibMetric",
    # "AnomalyScoreDistribution",
    # "BinaryPrecisionRecallCurve",
    # "create_anomalib_metric",
    # "Evaluator",
    "F1AdaptiveThreshold",
    # "F1Max",
    # "F1Score",
    # "ManualThreshold",
    "MinMax",
    # "PRO",
    # "PIMO",
    # "AUPIMO",
]