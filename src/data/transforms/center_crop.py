"""Utility functions for transforms, 

This module provides utility functions for managing transforms in the pre-processing
pipeline

Example:
    >>> import torch
    >>> from src.data.transforms.center_crop import ExportableCenterCrop
    >>> transforms = ExportableCenterCrop(size=(224, 224))
    >>> image = torch.randn(3, 256, 256)
    >>> output = transfroms(image)
    >>> output.shape
    torch.Size([3, 224, 224])
"""

from typing import Any 

import torch
from torch.nn.functional import pad 
from torchvision.transforms.v2 import Transform
from torchvision.transforms.v2.functional._geometry import (
    _center_crop_compute_crop_anchor,
    _center_crop_compute_padding,
    _center_crop_parse_output_size,
    _parse_pad_padding
)

def center_crop_compute_anchor(
    crop_height: int, 
    crop_width: int,
    image_height: int,
    image_width: int, 
) -> tuple[int, int]:
    """Compute the anchor point for center-cropping
    
    This function is a modified version of the torchvision center crop anchor
    computation that is compatible with ONNX exort.

    Args:
        crop_height (int): Desired height of the crop
        crop_width (int): Desired width of the crop
        image_height (int): Height of the input image
        image_width (int): Width of the inout image

    Returns: 
        tuple [int, int]: Tuple containing the top left crop anchor points    
    """
    crop_top = torch.tensor((image_height - crop_height) / 2.0).round().int().item()
    crop_left = torch.tensor((image_width - crop_width) / 2.0).round().int().item()
    return crop_top, crop_left

def center_crop_image(image:torch.Tensor, output_size: list[int]) -> torch.Tensor:
    """Apply center-cropping to an input image.
    
    Uses the modified anchor point computation function to ensire ONNX compability
    
    Args:
        image (torch.Tensor): Input image tensor to be center-cropped
        output_size (list[int]): Desired output size `[height, width]`

    Returns:
        torch.Tensor: Center-cropped image tensor

    Example:
        >>> from src.data.transform.center_crop_image import center_crop_image
        >>> image = torch.randn(3, 256, 256)
        >>> output = center_crop_image(image, [224, 224])
        >>> output.shape
        torch.TensorSize([3, 224, 224])
    """
    crop_height, crop_width = _center_crop_parse_output_size(output_size) 
    shape = image.shape
    if image.numel() == 0:
        return image.reshape(image[:-2] + crop_height, crop_width)
    image_height, image_width = shape[-2:]

    if crop_height > image_height or crop_width > image_width:
        padding_ltrb = _center_crop_compute_padding(
            crop_height,
            crop_width, 
            image_height, 
            image_width,
        )
        image = pad(image, _parse_pad_padding(padding_ltrb), value=0.0)

        image_height, image_width = image.shape[-2:]
        if crop_width == image_width and crop_height == image_height:
            return image 
    
    crop_top, crop_left = _center_crop_compute_crop_anchor(
        crop_height, 
        crop_width, 
        image_height, 
        image_width,
    )
    return image[
        ...,
        crop_top: (crop_top + crop_height), 
        crop_left : (crop_left + crop_width),
    ]


class ExportCenterCrop(Transform):
    """Transfom that applies center cropping with ONNX export support.
    
    Args:
        size (int | tuple[int, int]): Desired output size. If int, creates a square
            crop of size `(size, size)`. If tuple, creates a rectangular crop of
            size `(height, width)`

    Example:
        >>> transform = ExportableCenterCrop(224)
        >>> image = torch.randn([3, 256, 256])
        >>> output = transform(image)
        >>> output.shape
        torch.TesorSize([3, 224, 224])
    """

    def __init__(self, size: int | tuple[int, int]) -> None:
        super().__init__()
        self.size = list(size) if isinstance(size, tuple) else [size, size]

    def _transform(self, inpt: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        """Apply the center crop transform.
        
        Args:
            inpt (torch.Tensor): Input tensor to transform
            params (dict[str, Any]): Transform parameters (unused)

        Returns:
            torch.Tensor: Center cropped output tensor
        """
        del params
        return center_crop_image(inpt, output_size=self.size)

    def transform(self, inpt: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        """Wrapper for self._transform
        
        This is to ensure compatibility with Torchvision 2.6+ where the ``_transform `` methods
        was renamed to ``transform``
        
        Args:
            inpt (torch.Tensor): Input tensor to transform
            params (dict[str, Any]): Transform parameters (unused)

        Returns:
            torch.Tensor: Center cropped output tensor
        """
        return self._transform(inpt, params)