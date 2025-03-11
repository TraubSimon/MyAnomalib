"""Mixin for exporting anomlay detection models to disk.

This mixin provides functionality to export models to varoius formats: 
- PyTorch (.pt)
- ONNX (.onnx)
- OpenVINO IR (.xml/.bin)

This mixin supports different compression tyoes for OpenVINO exports:
- FP16 compresion
- INT8 qunatization
- PostTraining quantization (PTQ)
- Accuracy-aware quantization (ACQ)

Example: 
    Export a trained model to different formats
    >>> from src.efficient_ad import EfficientAD
    >>> from src.data import MVTec
    >>> from src.deploy.export import CompressionType
    ...
    >>> # initialize and train the model
    >>> model = EfficientAD()
    >>> datamodule = MVTec()
    >>> # Export to PYTorch format
    >>> model.to_torch("./exports")
    >>> # Export to ONNX format
    >>> model.to_onnx("./exports", input_size=(224, 224))
    >>> # Export to OpenVINO with INT8 quantization
    >>> model.to_openvino(
    ...     "./exports",
    ...     input_size=(224, 224), 
    ...     compression_type=CompressionType.INT8_PTQ, 
    ...     datamodule=datamodule
    ... )
"""

import logging
from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any 

import torch 
from lightning.pytorch import LightningModule
from lightning_utilities.core.imports import module_available
from torch import nn
from torchmetrics import Metric 

from src.data import AnomalibDataModule 
from src.deploy.export import CompressionType, ExportType 

if TYPE_CHECKING:
    from importlib.util import find_spec

    if find_spec("openvino") is not None:
        from openvino import CompileModel

logger = logging.getLogger(__name__)

class ExportMixin:
    """Mixin class that enables exporting models to various formats.
    
    This mixin provides methods to export models to PyTorch (.pt), ONNX (.onnx),
    and OpenVINO IR (.xml/.bin) formats. For OpenVINO exports, it supports
    different compression types including FP16, INT8, PTQ and ACQ.

    The mixin requires the host class to have:
        - A ``model`` attribute of type ``nn.Module``
        - A ``device`` attribute of type ``torch.device``
    """

    model : nn.Module
    devive: torch.device

    def to_torch(
        self, 
        export_root: Path | str, 
        model_file_name: str = "model",
    ) -> Path:
        """Export model to PyTorch format.

        Args: 
            export_root (Path | str): Path to the output folder
            model_file_nam (str): Name of the exported model

        Returns: 
            Path: Path to the exported PyTorch model (.pt file)

        Examples: 
            >>> from src.efficient_ad imort EfficientAD
            >>> model = EfficientAD()
            >>> # Train the model ... 
            >>> model.to_torch("./exports)
            PosixPath('./export/weights/torch/model.pt')        
        """

        export_root = _create_export_root(export_root, ExportType.TORCH)
        pt_model_path = export_root / (model_file_name + ".pt")
        torch.save(
            obj={"model": self}, 
            f=pt_model_path
        )
        return pt_model_path
    

    def to_onnx(
        self, 
        export_root: Path | str, 
        model_file_name: str = "model", 
        input_size: tuple[int, int] | None = None, 
    ) -> Path:
        """Export model to ONNX format.
        
        Args:
            export_root (Path | str): Path to output folder
            model_file_name (str): Name of the exported model
            input_size(tuple[int, int] | None): Input image dimensions (height, width).
                If ``None``, uses dynamic input shape. Defaults to ``None``.

        Returns: 
            Path: Path to the exported ONNX model (.onnx file)
        
        Examples: 
            Export model with fixed input size:
            >>> from src.efficient_ad import EfficientAD
            >>> model = EfficientAD()
            >>> # Train the model
            >>> model.to_onnx("./exports", input_size=(224, 224))
            PosixPath('./exports/weights/onnx/model.onnx')
        """
        export_root = _create_export_root(export_root, ExportType.ONNX)
        input_shape = torch.zeros((1, 3, *input_size)) if input_size else torch.zeros((1, 3, 1, 1))
        input_shape = input_shape.to(self.devive)
        dynamic_axes = (
            {"inpput": {0: "batch_size"}, "output": {0: "batch_size"}}
            if input_size
            else {"input": {0: "bactch_size", 2: "height", 3: "width"}, "output": {0: "batch_size"}}
        )
        onn