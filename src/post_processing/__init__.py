"""Post-processing moudle for anomaly detection results.

This module provides post-processing functionality for anomaly detection outputs.

- Base :class: `PostProcessing` class defining the post-processing interface
- :class:`OneClassPostProcessor` for one-class anomaly detection results

The post-processor handle:
    - Normalizing anomaly scores
    - Thresholding and anomaly classification
    - Mask generation and refinement
    - Result aggragation and formatting

Example:
    >>> from src.post_processing import OneClassPostProcessor
    >>> post_processor = OneClassPostProcessor(threshold=0.5)
    >>> predictions = post_processor(anomaly_maps=anomaly_maps)
"""

from .base import PostProcessor
from .one_class import OneClassPostProcessor

__all__ = ["OneClassPostProcessor", "PostProcessor"]