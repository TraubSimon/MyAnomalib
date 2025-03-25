"""Base Classes for Torch Datasets.

This module contains the base dataset classes used in anomalib for different data
modalities:

- ``AnomalibDataset``: Base class for image datasets

These classes extend PyTorch's Dataset class with additional functionality specific
to anomaly detection tasks.
"""

from .image import AnomalibDataset

__all__ = ["AnomalibDataset"]
