"""Anomalib Image Data Modules

This module contains data modules for loading and processing image datasets for
anomaly detection. The following data modules are available:

- ``MVTec``: MVTec Anoamly Detection Dataset
- ``Visa``: Visual Inspection for Steel Anomaly Dataset

Example:
    >>> from src.data import MVTec
    >>> datamodule = MVTec(
    ...     root="./datasets/MVTec",
    ...     category="bottle"
    ... )
"""

from enum import Enum 

from .mvtec import MVTec
from .visa import Visa 

class ImageDataFormat(str, Enum):
    """Supported Image Dataset Types.
    
    The following dataset formats are supported:
    - ``MVTEC``: MVTec Anoamly Detection Dataset
    - ``VISA``: Visual Inspection for Steel Anomaly Dataset 
    """
    MVTEC = "mvtec"
    VISA = "visa"

__all__ = ["MVTec", "Visa"]