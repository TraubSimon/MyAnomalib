"""Pre-processing module for anomaly detection piplines

This module provides functionality for preprocessing data before model training and 
inference through the :class:`PreProcessor` class


The pre-processor handles:
    - Applying transforms to data during different pipeline stages
    - Manage stage-speciofic tranfromations (train/test/val)
    - Integrationg with both PyTorch and Ligthening workflows

Example:
    >>> from src.pre_preocessing import PreProcessor
    >>> from torchvision.transforms.v2 import Resize
    >>> pre_processor = PreProcessor(transform=Resize(size(256, 256)))
    >>> transformed_bacth = pre_processor(batch)

The pre-processor is implemented as both a :class:`torch.nn.Module` and 
:class:`lightning.pytorch.Callback` to support inference and training workflows
"""

from pre_processing import PreProcessor
__all__ = ["PreProcessor"]