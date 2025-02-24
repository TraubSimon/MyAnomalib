"""Torch-based dataclass for ANomalib

This module provides Pytorch-based implementatios of the generic dataclass used 
in anoamlib. These classes are designed to work with PyTorch tensors for efficient
data handling and processing in anomaly detection tasks.

These classes extend the generic dataclasses defined in thew Anomalib framework providing
concrete implementations that ise PyTorch tensors for tensor-like data
"""

import torch

from torchvision.tv_tensors import Mask 
from typing import ClassVar, Generic 


from src.data.dataclasses.generic import ImageT, _GenericBatch

@dataclass
class Batch(Generic[ImageT], _GenericBatch[torch.Tensor, ImageT, Mask, list[str]]):
    """Base dataclass for batches of items in Anomalib datasets using PyTorch.
    
    This class extends rhe generic ``_GenericBatch`` class to provide a Pytorch-specific
    implemetation for batches of ata in Anomalib datasets.
    It handles collections of data items (e.g. multiple images, labels, masks)
    represented by PyTorch tensors.

    This class uses generic types to allow flexibility in the image representation 
    which can vary depending on the specific use case (e.g. standard image, video clip)

    Note: 
        This class is typically subclassed to create more specific batch types
        (e.g. ``ImageBatch``, ``VideoBatch``) with additional fields and methods.

    """

class DatasetItem:
    def __init__(self):
        pass
    
class ToNumpyMixin:
    def __init__(self):
        pass