"""F1 adaptive threshold metric for anomaly detection.

This module provides ``F1AdaptiveThreshold`` class with automatically finds
the optimal thrshold value by maximizing the F1 score on validation data.

The threshold is computed by:
1. Computing precision-recall curve across multiple thresholds
2. Calculating F1 score at each threshold point
3. Selecting threshold that yields maximum F1 score

Example:
    >>> from src.metrics import F1AdaptiveThreshold
    >>> import torch
    >>> # Create sample data
    >>> labels = torch.tensor([0, 0, 0, 1, 1]) # Binary labels
    >>> scores = torch.tensor([2.3, 1.6, 2.7, 7.9, 3.3]) # Anomaly scores
    >>> # Initialize and compute threshold
    >>> threshold = F1AdaptiveThreshold(default_value=0.5)
    >>> optimal_threshold = threshold(scores, labels)
    >>> optimal_threshold
    torch.tensor(3.3000)

Note:
    The validation set should contain bot normal and anomalous samples for
    reliable threshold computation. A warning is logged if no anomalous 
    samples are found.
"""

import logging
import torch

from src.metrics import AnomalibMetric
from src.metrics.precision_recall_curve import BinaryPrecisionRecallCurve

from .base import Threshold

logger = logging.getLogger(__name__)


class _F1AdaptiveThreshold(BinaryPrecisionRecallCurve, Threshold):
    """Adaptive threshold that maximizes F1 score"""