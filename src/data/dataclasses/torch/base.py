"""Torch-based dataclass for ANomalib

This module provides Pytorch-based implementatios of the generic dataclass used 
in anoamlib. These classes are designed to work with PyTorch tensors for efficient
data handling and processing in anomaly detection tasks.

These classes extend the generic dataclasses defined in thew Anomalib framework providing
concrete implementations that ise PyTorch tensors for tensor-like data
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import ClassVar, Generic, NamedTuple, TypeVar


import torch
from torchvision.tv_tensors import Mask 

from src.data.dataclasses.generic import ImageT, _GenericBatch, _GenericItem

NumpyT = TypeVar("NumpyT")


class InferencBatch(NamedTuple):
    """Batch for use in torch and inference models.
    
    Args:
        pred_score (torch.Tensor | None): Predicted anomaly scores.
            Defaults to `None`.
        pred_label (torch.Tensor | None): Predicted anomaly labels
            Defaults to `None`.
        anoamly_map (torch.Tensor | None): Generated anomaly maps.
            Defaulst to `None`.
        pred_mask (torch.Tensor | None): Predicted anomaly masks.
            Defaults to `None`.
    """
    pred_score: torch.Tensor | None = None
    pred_label: torch.Tensor | None = None
    anomaly_map: torch.Tensor | None = None
    pred_mask: torch.Tensor | None = None

@dataclass
class ToNumpyMixin(Generic[NumpyT]):
    """Mixin for converting torch-based dataclasse to numpy
    
    This mixin provides functionality to convert PyToorch tensor data to numpy 
    arras. It requuires the subclass to define a `Numpy_class` attributes specifying
    the corresponding numpy-based class.

    Examples:
        >>> from src.dataclasses.numpy import NumpyImageItem
        >>> @dataclass
        ... class TorchImageItem(ToNumpyMixin[NumpyImageItem]):
        ...     numpy_class = NumpyImageItem
        ...     image: torch.Tensor
        ...     gt_labe: torch.Tensor
        ...
        >>> torch_item = TorchImageItem(
        ...     image=torch.randn(3, 224, 224), 
        ...     gt_label=torch.tensor(1),
        ... )
        >>> numpy_item = torch_item.to_numpy()
        >>> isinstance(numpy_item, NumpyImageItem)
        True    
    """

    numpy_class: ClassVar[Callable]

    def __init_subclass__(cls,  **kwargs) -> None:
        """Ensure that the subclass has the required attributes.
        
        Args: 
            **kwargs: Additional keyword arguments passed to parent class

        Raises: 
            AttributeError: If the subclass does not define `numpy_class`.        
        """
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "numpy_class"):
            msg = f"{cls.__name__} must have a `numpy_class` attribute."
            raise AttributeError(msg)
        
    def to_numpy(self) -> NumpyT:
        """Convert the batch to a NumpyBatch object. 
        
        Returns:
            NumpyT: the converted numpy batch object.        
        """
        batch_dict = asdict(self)
        for key, value in batch_dict.items():
            if isinstance(value, torch.Tensor):
                batch_dict[key] = value.cpu().numpy()
        return self.numpy_class(
            **batch_dict
        )

@dataclass
class DatasetItem(Generic[ImageT], _GenericItem[torch.Tensor, ImageT, Mask, str]):
    """Base dataclass for individual items in anomalib datasets using PyTorch.
    
    This class extends the generic `_GenericItem` class to 
    provude a PyTorch specific implementation for single data
    items in Anomalib datasets. 
    It handels various types of data (e.g. images, labels, masks)
    represeted as PyTorch tensors. 

    The class uses generic types to allow flexibility in the image 
    representation, which can vary depending on the specific use
    case (e.g. standard images, video clips).

    Note:
        This class is typicalley subclassed to create more specific item
        types (e.g. `ImageItem`, `VideoItem`) with additional fields and
        methods.
    """

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

