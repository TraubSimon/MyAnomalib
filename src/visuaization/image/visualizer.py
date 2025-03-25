"""Image visualization module for anomaly detection.

This module provides the ``ImageVisualizer`` class for visualizing images and their
associated anomaly detection results. The key components include:

    - Visualization of individual fields (images, masks, anomaly maps)
    - Overlay of multiple fields
    - Configurable visualization parameters
    - Support for saving visualizations

Example:
    >>> from src.visualization.image import ImageVisualizer
    >>> # Create visualizer with default settings
    >>> visualizer = ImageVisualizer()
    >>> # Generate visualization
    >>> vis_result = visualizer.visualize(image=img, pred_mask=mask)

The module ensures consistent visualization by:
    - Providing standardized field configurations
    - Supporting flexible overlay options
    - Handling text annotations
    - Maintaining consistent output formats

Note:
    All visualization functions preserve the input image format and dimensions
    unless explicitly specified in the configuration.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any 

# only import types during type checking to avoid circular imports
if TYPE_CHECKING:
    from lightning.pytorch import Trainer

    from src.data import ImageBatch
    from src.models import AnomalibModule