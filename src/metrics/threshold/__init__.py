"""Thresholding merics for anomaly detection.

This module provides varoius thresholding techniques to convert anomaly 
scores to binary predictions.

Available thresholds:
    - ``BaseThreshold``: Abstract base class for implementing threshold metrics
    - ``Threshold``: Generic threshold class that can be initialized with a value
    - ``F1AdaptiveThreshold``: Automatically finds optimal thrshold by maximizing
        F1 score
    - ``ManualThreshold``: Allows manual setting of threshold value

Example: 
    >>> from src.metrics.threshold import ManualThreshold
    >>> threshold = ManualThreshold(0.5)
    >>> predictions = threshold(anomaly_scores=[0.1, 0.6, 0.3, 0.9])
    >>> predictions
    torch.tensor([0, 1, 0, 1])
"""

from .base import BaseThreshold, Threshold
from .f1_adaptive_threshold import F1AdaptiveThreshold
from .manual_threshold import ManualThreshold

__all__ = ["BaseThreshold", "Threshold", "F1AdaptiveThreshold", "ManualThreshold"]