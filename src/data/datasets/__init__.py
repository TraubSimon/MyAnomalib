"""PyTorch Dataset implementations for anomly detection

This module provides dataset implmentations for various 
anomly datection tasks:

Base Classses:
    - ``AnomalibDataset``: Base class for all Anomalib datasets

Image Datasets:
    - ``MVTecDataset``: MVTec AD dataset with industrial objects
    - ``VisADataset``: Visual Inspection of Surface Anomalies Dataset

Example:
    >>> from src.data.datasets import MVTecDataset
    >>> dataset = MVTecDataset(
    ...     root="./datasets/MVTec", 
    ...     category="bottle", 
    ...     split="train",
    ... )
"""

from .base import AnomalibDataset
from .image import MVTecDataset, VisaDataset

__all__ = [
    # Base
    "AnomalibDataset",
    # Imgae
    "MVTecDataset", 
    "VisaDataset",
]