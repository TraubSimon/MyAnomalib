"""Visualization module for anomaly detection

    This module provides utilities for visualizing anomaly detection results.
    - Base ``Visualizer```class defining the visualization interface
    - ``ImageVisualizer`` class for image visualization
    - Functions for visualizing anomaly maps and segmentation masks
    
"""
from .base import Visualizer
from .image import ImageVisualizer, visualize_anomaly_map, visualize_mask

__all__ = [
    # Base visualizer class
    "Visualizer",
    # Image visualizer class
    "ImageVisualizer",
    # Image visualization functions
    "visualize_anomaly_map",
    "visualize_mask",
]