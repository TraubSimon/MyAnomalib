"""PyTorch Dataset implementations for anomaly detection in images.

This module provides dataset implementations for various image anomaly detection
datasets:

- ``MVTecDataset``: MVTec AD dataset with industrial objects
- ``VisaDataset``: Visual Inspection of Surface Anomalies dataset

Example:
    >>> from anomalib.data.datasets import MVTecDataset
    >>> dataset = MVTecDataset(
    ...     root="./datasets/MVTec",
    ...     category="bottle",
    ...     split="train"
    ... )
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from .mvtec import MVTecDataset
from .visa import VisaDataset

__all__ = [
    "MVTecDataset",
    "VisaDataset",
]
