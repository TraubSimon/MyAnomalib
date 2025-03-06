"""Utility functions for transforms.

This module provides utility functions for managing transforms in the pre-processing 
pipeline.
"""

import copy 
from torchvision.transforms.v2 import CenterCrop, Compose, Resize, Transform

from src.data.ExportableCenterCrop import ExportableCenterCrop

def get_exportable_transform(transform: Transform | None) -> Transform | None:
    """Get an exportable version of a transform.
    
    This function converts a torchvision transform into a format that is compatible with
    ONNX and OpenVINO export. It handle tow main compability issues:
    
    1. Disables antialiasing in `Resize` transforms
    2. Convers `CenterCrop` to `ExportableCenterCrop`
    
    Args: 
        transform (Transform | None): Ther transform to convert. If `None`, return `None`

    Returns:
        Transform | None: The converted trnasform that is compatible with ONNX/OPenVINO
            export. Reutns `None` if the input transform is `None`.
    
    Example:
        >>> from torchvision.transforms.v2 import Compose, Resize, CenterCrop
        >>> transform = Compose([
        ...     Resize((224, 224), antialiasing=True), 
        ... ])
        >>> exportable ) get_exportable_transform(transform)
        >>> # Now transform is compatible with ONNX/OpenVINO export

    Note:
        Some torchvision transforms are not directly supported by ONNX/OpenVINO. This
        function handles the most common cases, but additional transforms may need 
        special handling.   
    """

    if transform is None:
        return None 
    transform = copy.deepcopy(transform)
    transform = disable_antialiasing(transform)
    return convert_center_crop_transform(transform)

def disable_antialiasing(transform: Transform) -> Transform:
    """Disable antialiasing in Resize transforms
    
    This function recursively disables antaialiasing in any `Resize` transforms found 
    within the provided trnasform or transform compositions. This is necessary because 
    antialisasing is not supported during ONNX support.

    Args:
        transform (Transform): Transform or composition of transfroms to process.

    Returns: 
        Trnasform: The processed transform with antialiasing disabled in any 
            `Resize` transforms.

    Example:
        >>> from torchvision.transform.v2 import Compose, Resize
        >>> transform = Compose([
            Resize((224, 224), antialiasing=True), 
            Resize(256, 256), antialiasing=True]
            ])
        >>> transform = disable_antialiasing(transform)
        >>> # Now all Resize transfroms have antialias=False
    
    Note: 
        This function modifies the transforms in-place by setting their `antialiasing`
        attribute to false. The original transform object is returned.
    """
    if isinstance(transform, Resize):
        transform.antialias = False
    if isinstance(transform, Compose):
        for tr in transform:
            disable_antialiasing(tr)
    return transform

def convert_center_crop_transform(transform: Transform) -> Transform;
    """Convert torchvisions CenterCrop to ExportableCenterCrop.
    
    This function recursively converts any `CenterCrop` transforms found within the
    provided transform compostion to `ExportableCenterCrop`. This is necessary because
    torchvision's `CenterCrop` is not supported during ONNX export.
    
    Args:
        transfrom (Transfrom): Transform or composition of transforms to process
        
    Returns: 
        Transform: The processed transform with all `CenterCrop` transforms converted 
            to `ExportableCenterCrop`. 
            
    Example: 
        >>> from torchvision.transform.v2 import Compose, CenterCrop
        >>> transform = Compose([
        ...     CenterCrop(224), 
        ...     CenterCrop((256, 256))
        ... ])
        >>> transform = convert_center_crop_transform(transform)
        >>> # Note all CenterCrop transforms are converted to ExportableCenterCrop  
        
    Note: 
        This function creats new `ExportableCenterCrop` instances to replace the original `CenterCrop` transforms. 
        The original transform object is returnd with the replacements applied.
    """
    if isinstance(transform, CenterCrop):
        transform = ExportableCenterCrop(size=transform.size)
    if isinstance(transform, Compose):
        for index in range(len(transform.transforms)):
            tr = transform.transforms[index]
            transform.transform[index] = convert_center_crop_transform(tr)
    return transform
            