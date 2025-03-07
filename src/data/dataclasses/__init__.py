"""Anomalib dataclasses. 

This module provides a collection of dataclasses used throuout the Anomalib
library for representing and managing various types of data related to anomaly
detection tasks.

The dataclasses are organized into two main categories:
1. Numpy-based dataclasses for handling numpy array data
2. Torch-based dataclasses for handling PyTorch tensor data.

Key components
--------------

Numpy Dataclasses
~~~~~~~~~~~~~~

- :classs:`NumpyImageItem`: Single image iteam as numpy arrays
    - Data shape `(H, W, C)` or `(H, W)` for grayscale
    - Labels: Binary classification: (o: normal, 1: anomalous)
    - Masks: Binary segmentation masks `(H, W)`

- :class: `NumpyImageBatch`: Batch of image data as numpy arrays
    - Data shape: `(N, H, W, C)` or `(N, H, W)` for grayscale
    - Labels: `(N,)` binary labels
    - Masks: `(N, H, W)` binary masks

Torch Dataclasses
~~~~~~~~~~~~~~~

- :class: `Batch`: Base class for torch-based batch data
- :class: `DatasetItem`: Base class for torch-based dataset items
- :class: `ImageItem`: Single image as torch tensors
    - Data shape: `(C, H, W)`
- :class: `ImageBatch`: Batch of images
    - Data shape: `(N, C, H, W)`
- :class: `InferenceBatch`: Specialized batch for inference results
    - Predictions: Score, labels, anomaly maps and masks

These dataclasses provide a structured way to handle various types of data
in anomaly detection tasks, ensuring type consistency and easy data manipulation
across different components of the anomalib library

Example:
----
>>> from src.data.dataclasses import ImageItem
>>> import torch
>>> item = ImageItem(
...     image=torch.rand(3, 224, 224), 
...     gt_label=torch.tensor(0),   
...     image_path="path/to/image.jpg"
... )
>>> item.image.shape
torch.Size([3, 224, 224])
"""

from .numpy import NumpyImageBatch, NumpyImageItem
from .torch import (
    Batch, 
    DatasetItem, 
    ImageBatch, 
    InferenceBatch, 
)

__all__ = [
    # Numpy
    "NumpyImageItem", 
    "NumpyImageBatch",
    # Torch
    "DatasetItem",
    "Batch",
    "InferenceBatch", 
    "ImageItem",
    "ImageBatch", 
]