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
    """Adaptive threshold that maximizes F1 score
    This class computes and stores the optmial threshold for converting
    anomaly scores to binary predictions by maximizing the F1 score on 
    validation data.

    Example:
        >>> from src.matrics import F1AdaptiveThreshold
        >>> import torch
        >>> # Create validation data
        >>> labels = torch.tensor([0, 0, 1, 1]) # 2 normal, 2 anomalous
        >>> scores = torch.tensor([0.1, 0.2, 0.8, 0.9]) # Anomaly scores
        >>> # INitialize thresholds
        >>> threshold = F1AdaptiveThreshold()
        >>> # Compute optimal threshold
        >>> optimal_value = threshold(scores, labels) 
        >>> print(f"Optimal threshold: {optimal_value:.4f}")
        Optimal threshold: 0.5000
    """

    def compute(self) -> torch.Tensor:
        """Compute optimal threshold by maximizing F1 score.

        Calculates precision-recall curve corresponding thresholds, then
        finds the threshold that maximizes F1 score.

        Returns: 
            torch.Tensor: Optimal threshold value.

        Warning:
            If validation set contains no anomalous samples, the threshold woll
            defaiöt to the maximum anomaly score, which may lead to poor performance.
        """
        precision: torch.Tensor
        recall: torch.Tensor
        thresholds: torch.Tensor

        if not any(1 in batch for batch in self.target):
            msg = ("The validation seth does not contain any anomalous images as a "
                   "result, the adaptive threshold will take the value of the "
                   "highest anoamly score observed in the normal validation images, "
                   "which maay lead to poor predictions. For a more reliables "
                   "adaptive threshold computation please add some anomalous"
                   "images to the validation set.")
            logging.warning(msg)

        precision, recall, thresholds = super().compute()
        f1_score = (2 * precision * recall) / (precision + recall + 1e-10)

        # account for special case where recall is 1.0 even for the highest threshold.
        # In this case 'thresholds' will be a scalar. 
        return thresholds if thresholds.dim() == 0 else thresholds[torch.argmax(f1_score)]
    
class F1AdaptiveThreshold(AnomalibMetric, _F1AdaptiveThreshold):  # tyoe: ignore[misc]
    """Wrapper to add Anomalib Metric functionality to F1AdaptiveThreshold metric.""" 