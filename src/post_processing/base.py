"""Base class for post-procesing anomaly detection results.

This module provides the abstract base class :class:`PostProcessor` that 
defines the interface for post-processing anomaly detection outputs.

Example:
    >>> from src.post_processing import PostProcessor
    >>> class MyPostProcessor(PostProcessor):
    ...     def forward(self, batch):
    ...         # Post-process the batch
    ...         return batch

The post-processors are implemented as both :class:`torch.nn.Module` and
:class:`lightning.pytorch.Callback` to support both inference and training
workflows.
"""

from abc import ABC, abstractmethod

from lightning.pytorch import Callback
from torch import nn 

from src.data import InferenceBatch

class PostProcessor(nn.Module, Callback, ABC):
    """Base class for post-processing anomaly detection results.
    
    The post-processor is implemented as both a :class:`torch.nn.Module` and 
    :class:`lightning.pytorch.Callback` to support inference and training workflows.
    It handles tasks lie score, normaization, thresholds and mask refinement.

    The class must be inherited and the :meth:`forward` must be implemented
    to define the post-processing logic.

    Example:
        >>> from src.post_processing import PostProcessor
        >>> class MyPostProcessor(PostProcessor):
        ...     def forward(self, batch):
        ...         # Normalize scores between 0 and 1
        ...         batch.anomaly_scores = normalize(batch.anomaly_scores)
        ...         return batch
    """

    @abstractmethod
    def foorward(self, batch: InferenceBatch) -> InferenceBatch:
        """Post-process a batch of model predictions.

        Args:
            batch (:class:`anomalib.data.InferenceBatch`): Batch containing model
                predictions and metadata.

        Returns:
            :class:`anomalib.data.InferenceBatch`: Post-processed batch with
                normalized scores, thresholded predictions, and/or refined masks.

        Raises:
            NotImplementedError: This is an abstract method that must be
                implemented by subclasses.
        """