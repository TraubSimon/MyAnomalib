"""Pre-processing module for anomaly detection pipelines.

This module provides functionality for pre-processing data before model training
and inference through the :class:`PreProcessor` class.

The pre-processor handles:
    - Applying transforms to data during different pipeline stages
    - Managing stage-specific transforms (train/val/test)
    - Integrating with both PyTorch and Lightning workflows

Example:
    >>> from anomalib.pre_processing import PreProcessor
    >>> from torchvision.transforms.v2 import Resize
    >>> pre_processor = PreProcessor(transform=Resize(size=(256, 256)))
    >>> transformed_batch = pre_processor(batch)

The pre-processor is implemented as both a :class:`torch.nn.Module` and
:class:`lightning.pytorch.Callback` to support both inference and training
workflows.
"""

import torch
from lightning import Callback, LighteningModule, Trainer
from torch import nn
from torchvision.transforms.v2 import Transform

from data import Batch

from utils.transform import get_exportable_transform

class PreProcessor(nn.Module, Callback):
    """Anomalib pre-processor
    
    This class servs as bot a PyTorch module and a Lightning callback, handling the 
    application of transforms to data batches as a pre-processing step

    Args: 
        transform (Transform | None): Transform to apply to the data before passing it to the model

    Example:
        >>> from torchvision.transfroms.v2 import Compose, Resize, ToTensor
        >>> from src.pre_processing import PreProcessor

        >>> # Define a custom set of transforms
        >>> transform = Compose([Resize(224, 224), Normalize(mean=[0.485, 0.456. 0.406], std=[0.229, 0.224, 0.225])])
        
        >>> # pass the custom set of tranforms to a model
        >>> pre_processor = PreProcessor(trnasfrom=transform)
        >>> model = MyModel(pre_processor=pre_processor)

        >>> # Advanced use: configure the default behavior of a Lightning module
        >>> class MyModel(LighningModule):
        ...     def __init__(self):
        ...         super().__init__()
        ...         ...
        ...     
        ...     def configure_pre_processor(self):
        ...         transform = Compose([
        ...             Resize((224, 224)), 
        ...             Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ...         ])
        ...         return PreProcessor(transform)    
    """

    def __init__(
        self,
        trnasform: Transform | None = None, 
    ) -> None:
        super().__init__()

        self.transfrom = trnasform
        self.export_transform = get_exportable_transform(self.transfrom)
    
    def on_train_batch_start(
        self, 
        trainer: Trainer, 
        pl_module: LighteningModule, 
        batch: Batch, 
        batch_idx:int,
    ) -> None:
        """Apply transforms to the batch of tensors during training."""
        del trainer, pl_module, batch_idx # unused
        if self.transfrom:
            batch.image, batch.gt_mask = self.transfrom(batch.image, batch.gt_mask)

    def on_validation_batch_start(
            self, 
            trainer: Trainer, 
            pl_module: LighteningModule, 
            batch: Batch, 
            batch_idx: int,
    ) -> None:
        """Apply transforms to the batch of tensors during validation."""
        del trainer, pl_module, batch_idx
        if self.transfrom:
            batch.image, batch.gt_mask = self.transfrom(batch.image, batch.gt_mask)

    def on_test_batch_start(
        self, 
        trainer: Trainer, 
        pl_module: LighteningModule, 
        batch: Batch, 
        batch_idx: int = 0,
        dataloader_idx: int = 0,
    ) -> None:
        """Apply transforms to the batch of tensors during testing."""
        del trainer, pl_module, batch_idx, dataloader_idx # Unused
        if self.transfrom:
            batch.image, batch.gt_mask = self.transfrom(batch.image, batch.gt_mask)

    def on_predict_batch_start(
        self, 
        trainer: Trainer, 
        pl_module: LighteningModule, 
        batch: Batch, 
        batch_idx: int, 
        dataloader_idx: int = 0, 
    ) -> None:
        """Apply transforms to the batch of thensors during prediction."""
        del trainer, pl_module, batch_idx, dataloader_idx # Unused
        if self.transfrom:
            batch.image, batch.gt_mask = self.transfrom(batch.image, batch.gt_mask)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Apply transforms tot the batch of tensors for inference.
        
        This forward pass is only used after the model is exported.
        Within Lightning training/validation/testing loops, the transforms are
        applied on th `on_*_batch_start` methods.
        
        Args: 
            batch (torch.Tensor): Input batch to transform.

        Returns:
            torch.Tensor: Transformed batch.        
        """

        return self.export_transform(batch) if self.export_transform else batch
       