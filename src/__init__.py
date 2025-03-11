"""Anomalib library for research and benchmarking.

This library provides tools and utilities for anomaly detection research and
benchmarking. The key components include:

    - EfficientAD anomaly detection model
    - Standardized training and evaluation pipelines
    - Support for various data formats and tasks
    - Visualization and analysis tools
    - Benchmarking utilities

Example:
    >>> from src.efficient_ad import EfficientAd
    >>> # Create and train model
    >>> model = EfficientAd()
    >>> model.train(train_dataloader)
    >>> # Generate predictions
    >>> predictions = model.predict(test_dataloader)

The library supports:

Note:
    The library is designed for both research and production use cases,
    with a focus on reproducibility and ease of use.
"""
