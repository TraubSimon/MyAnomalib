"""Base classes for thresholding metrics.

This module provides base classes for implementeing threshold based metrics
for anomaly detection. The main classes are:

- `Threshold`: Abstract base class for all threshold metrics
- `BaseThreshold`: Deprecated alias for `Threshold` class

Example:
    >>> from src.metrics.threshild import Threshold
    >>> class MyThreshold(Threshold):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.add_state("scores", default=[])
    ...
    ...     def update(self, scores):
    ...         self.scores.append(scores)
    ...
    ...     def compute(self):
    ...         return torch.tensor(0.5)
    ... threshold = MyThreshold()
    ... threshold.update(torch.tensor([0.1, 0.9]))
    ... threshold.compute()
    torch.tensor(0.5)
"""

import warnings

import torch
from torchmetrics import Metric

class Threshold(Metric):
    """Absteact base class for thresholding metrics.
    
    This class serves as the foundation for implementing threshold-based metrics in 
    anomaly detection. It inherits from `torchmetrics.Metric` and defines a common 
    interface for threshold computation and state updates.

    Subclasses must implement:
        - `compute()`: Calculates and returns the threshold value
        - `update()`: Update internal state with new data

    Example:
        >>> class MyThreshold(Threshold):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.add_state("scores", default=[])
        ...     def update(self, scores):
        ...         self.scores.append(scores)
        ...     def compute(self):
        ...         return torch.tensor(0.5)
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the threshold metric.
        Args: 
            **kwargs: Keyword arguments passed to parant `Metric` class.
        """
        super().__init__(**kwargs)

    def compute(self) -> torch.Tensor:  # noqa: PLR6301
        """Computes the threshold value.
        
        Returns:
            torch.Tensor: Optimal threshold value
        
        Raises: 
            NotImplementedError: If not implemented by subclass,
        """
        msg = "Subclass if Threshold must implment the `compute()` method"
        raise NotImplementedError(msg)
    
    def update(self, *args, **kwargs) -> None:  # noqa: ARG002, PLR6301
        """Update the metric state with new data.

        Args:
            *args: Positional arguments specific to subclass implementation.
            **kwargs: Keyword arguments specific to subclass implementation.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        msg = "Subclass of Threshold must implement the update method"
        raise NotImplementedError(msg)