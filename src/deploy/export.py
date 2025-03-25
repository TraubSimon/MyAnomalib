"""Utilities for optimization and OpenVINO conversion.

This module provides functionality for exporting and optimizing anomaly detection
models to different formats like ONNX and PyTorch.

Example:
    Export a model to ONNX format:

    >>> from src.deploy import ExportType
    >>> export_type = ExportType.ONNX
    >>> export_type
    'onnx'
"""



import logging
from enum import Enum

logger = logging.getLogger("anomalib")


class ExportType(str, Enum):
    """Model export type.

    Supported export formats for anomaly detection models.

    Attributes:
        ONNX: Export model to ONNX format
        OPENVINO: Export model to OpenVINO IR format
        TORCH: Export model to PyTorch format

    Example:
        >>> from anomalib.deploy import ExportType
        >>> export_type = ExportType.ONNX
        >>> export_type
        'onnx'
    """

    ONNX = "onnx"
    TORCH = "torch"

