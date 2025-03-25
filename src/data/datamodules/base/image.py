"""Base Anomalib data module.

This module provides the base data module class used across Anomalib. It andles 
dataset splitting, validation set creation and dataloader configuration.

The module contains:
    - :class:`AnomalibDataModule`: Base class for all Anomalib data modules.

Example:
    Crea a datamodule from a config file:
    >>> from src.data import AnomalibDataModule
    >>> data_config = "examples/configs/data/mvtec.yaml"
    >>> datamodule = AnomalibDataModule.from:config(config_path=data_config)

    Override config with additional arguments
    >>> override_kwargs = {data.train_batch_size": 8}
    >>> datamodule = AnomalibDataModule.from_config(
    ...     config_path=data_config, 
    ...     **override_kwargs
    ... )
"""
import copy
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from pandas import DataFrame

from lightning.pytorch import LightningDataModule
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.v2 import Compose, Resize, Transform


from src.data.utils import TestSplitMode, ValSplitMode, random_split, split_by_label
from src.data.datasets.base.image import AnomalibDataset
from src.data.transforms.utils import extract_transform_by_type
from src.utils.attr import get_nested_attr 



logger = logging.getLogger(__name__)

class AnomalibDataModule(LightningDataModule, ABC):
    """Base Anomalib data module
    
    This class extends PyTorchs Ligthning's ``LightningDataModule`` to provide
    common functinality for anomaly detection datasets.

    Args: 
        train_batch_size (int): Batch size used by the train dataloader.
        eval_batch_size (int): Batch size used by the test dataloaders.
        num_workers (int): number of workers used by the train, val and test dataloaders.
        train_augmentations (Transform | None): Augmentations to apply to the training images
            Defaults to ```None``
        test_augmentations (Transform | None): Augmentations to apply to the test images
            Defaults to ```None``
        augmentations (Transform | None): General Augmentations to apply if stage-specific
            Defaults to ```None``
        val_split_mode (ValSplitMode | str): Method to obtain validation set.
            Options:
                - ``none``: No validation set
                - ``same_as_test``: Use test set as validation
                - ``from_test``: Sample from test set
                - ``synthetic``: Generate synthetic anomalies
        val_split_ratio (float): Fraction of data to use for validation
        test_split_mode (TestSplitMode | str | None): Method to obtain test set.
            Options:
                - ``none``: No test split
                - ``from_dir``: Use separate test directory
                - ``synthetic``: Generate synthetic anomalies
            Defaults to ``None``.
        test_split_ratio (float | None): Fraction of data to use for testing.
            Defaults to ``None``.
        seed (int | None): Random seed for reproducible splitting.
            Defaults to ``None``.
    """


    def __init__(
        self, 
        train_batch_size: int, 
        eval_batch_size: int,
        num_workers: int,
        train_augmentations: Transform | None = None, 
        val_augmentations: Transform | None = None, 
        test_augmentations: Transform | None = None, 
        augmentations: Transform | None = None, 
        val_split_mode: ValSplitMode | str | None = None, 
        val_split_ratio: float | None = None, 
        test_split_mode: TestSplitMode | str | None = None, 
        test_split_ratio: float | None = None, 
        seed: int | None = None,
        ) -> None:
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.test_split_mode = TestSplitMode(test_split_mode) if test_split_mode else TestSplitMode.NONE
        self.test_split_ratio = test_split_ratio or 0.5
        self.val_split_mode = ValSplitMode(val_split_mode) if val_split_mode else ValSplitMode.NONE
        self.val_split_ratio = val_split_ratio  or None
        self.seed = seed 

        self.train_augmentations = train_augmentations or augmentations
        self.val_augmentations = val_augmentations or augmentations
        self.test_augmentations = test_augmentations or augmentations 

        self.train_data: AnomalibDataset
        self.val_data: AnomalibDataset
        self.test_data: AnomalibDataset

        self._smaples: DataFrame | None = None 
        self._category: str = "" 
        self._is_setup = False # flag to track if setup has already been called

    @property
    def name(self) -> str:
        """Name of the datamodule
        
        Returns:
            str: Class name of the data module
        """
        return self.__class__.__name__
   
    
    def setup(self, stage: str | None = None) -> None:
        """Set up train validation and test data
        
        This method handles the data splitting logic based on the 
        configured modes.

        Args:
            stage (str | None): Current stage (fit/test/validate/predict)
                Defaults to ``None``
        """
        has_subset = any(hasattr(self, subset) for subset in ["train_data", "val_data", "test_data"])
        if not has_subset or not self._is_setup:
            self._setup(stage)
            self._create_test_split()
            self._create_val_split()
            if isinstance(stage, TrainerFn):
                # oly set the flag if called from trainer
                self._is_setup = True 
        self._update_augmentations()
    
    def _update_augmentations(self) -> None:
        """Update the augmentations for each subset"""
        for subset_name in ["train", "val", "test"]:
            subset = getattr(self, f"{subset_name}_data", None)
            augmentations = getattr(self, f"{subset_name}_augmentations", None)
            model_transform = get_nested_attr(self, "trainer.model.pre_processor.transform")
            if subset and model_transform:
                self._update_subset_augmentations(subset, augmentations, model_transform)
    @staticmethod
    def _update_subset_augmentations(
        dataset: AnomalibDataset, 
        augmentations: Transform | None, 
        model_transform: Transform, 
    ) -> None:
        """Update the augmentations of the dataset.
        
        The method passes the user-specified augmentations to a dataset subset. If the model conatains 
        a Resize transform, it wall be appended to the augmentations. This will ensure that resizing takes
        place before collating, which reduces the usage of shared memory by the Dataloader workers.

        Args:
            dataset (AnomalibDataset): Dataset to update.
            augmentations (Transform | None): Augmentations to apply to the dataset
            model_transform (Transform): Transform object from the model PreProcessor.
        """
        model_resizes = extract_transform_by_type(model_transform, Resize)

        if model_resizes:
            model_resize = model_resizes[0]
            for aug_resize in extract_transform_by_type(augmentations, Resize):
                if model_resize.size != aug_resize.sze:
                    msg = f"Conflicting resize shapes found between augmentations and model transforms. You are using \
                        a Resize transform in your input data augmentations. Please be aware that the model also \
                        applies a Resize transform with a different output size. The final effective input size as \
                        seen by the model will be determined by the model transforms, not the augmentations. To change \
                        the effective input size, please change the model transforms in the PreProcessor module. \
                        Augmentations: {aug_resize.size}, Model transforms: {model_transform.size}"
                    logger.warning(msg)
                if model_resize.interpolation != aug_resize.interpolation:
                    msg = f"Conflicting interpolation method found between augmentations and model transforms. You are \
                        using a Resize transform in your input data augmentations. Please be aware that the model also \
                        applies a Resize transform with a different interpolation method. Using multiple interpolation \
                        methods can lead to unexpected behaviour, so it is recommended to use the same interpolation \
                        method between augmentations and model transforms. Augmentations: {aug_resize.interpolation}, \
                        Model transforms: {model_resize.interpolation}"
                    logger.warning(msg)
                if model_resize.antialias != aug_resize.antialias:
                    msg = f"Conflicting antialiasing setting found between augmentations and model transforms. You are \
                        using a Resize transform in your input data augmentations. Please be aware that the model also \
                        applies a Resize transform with a different antialising setting. Using conflicting \
                        antialiasing settings can lead to unexpected behaviour, so it is recommended to use the same \
                        antialiasing setting between augmentations and model transforms. Augmentations: \
                        antialias={aug_resize.antialias}, Model transforms: antialias={model_resize.antialias}"
                    logger.warning(msg)
        
            # append model resize to augmentations
            if isinstance(augmentations, Resize):
                augmentations = model_resize
            elif isinstance(augmentations, Compose):
                augmentations = Compose([*augmentations.transform, model_resize])
            elif isinstance(augmentations, Transform):
                augmentations = Compose([augmentations, model_resize])
            elif augmentations is None:
                augmentations = model_resize 
        
        dataset.augmentations = augmentations
    
    @abstractmethod
    def _setup(self, _stage: str | None = None) -> None:
        """Set up th datasets and perform dynamic subset splitting.
        
        This method should be implemented by subclasses to define dataset-specific 
        setup logic.

        Note: 
            The ``stage`` argument is not used since all subsets are created on first 
            call to accomodate validation set extraction from test set.

        Args: 
            _stage (str | None): Current stage (unused)
                Deaults to ``None``

        Raises: 
            NotImplementedError: When not implemented by subclasses
        """
        raise NotImplementedError
    
    @property
    def category(self) -> str:
        """Get dataaset category name.
        
        Returns:
            str: Name of the current category
        """
        return self._category
    
    @category.setter 
    def category(self, category: str) -> None:
        """Set dataset category name.
        
        Args: 
            category (str): Category name to set
        """
        self._category = category
    
    def _create_test_split(self) -> None:
        """Create the test split based on cogigured mode
        
        This handles splitting normal/anomlous samples and optionally
        creating sythentic anomalies.
        """
        if self.test_data.has_normal:
            # split test data into normal and anomalous
            normal_test_data, self.test_data = split_by_label(self.test_data)
        elif self.test_split_mode != TestSplitMode.NONE:
            # sample normal images from training set if none provided
            logger.info(
                "No normal test images found. Sampling from trainng set using ration of %0.2f",
                self.test_split_ratio
            )
            if self.test_split_ratio is not None:
                self.train_data, normal_test_data = random_split(
                    self.train_data, 
                    self.test_split_ratio, 
                    seed=self.seed
                )
        
        if self.test_split_mode == TestSplitMode.FROM_DIR:
            self.test_data += normal_test_data
        elif self.test_split_mode != TestSplitMode.NONE:
            msg = f"Unsupported Test Split Mode: {self.test_split_mode}"
            raise ValueError(msg)
    
    def _create_val_split(self) -> None:
        """Create validation split based on configured mode.
        
        This handles sampling from tain/test sets
        """
        if self.val_split_mode == ValSplitMode.FROM_TRAIN:
            # randoly sample from train set
            self.train_data, self.val_data = random_split(
                self.train_data,
                self.val_split_ratio,
                label_aware=True, 
                seed=self.seed, 
            )
        elif self.val_split_mode == ValSplitMode.FROM_TEST:
            # randomly sample from test set
            self.test_data, self.val_data = random_split(
                self.test_data, 
                self.val_split_ratio,
                label_aware=True, 
                seed=self.seed,
            )
        elif self.val_split_mode == ValSplitMode.SAME_AS_TEST:
            # equal to test set
            self.val_data = copy.deepcopy(self.test_data)
        elif self.val_split_mode != ValSplitMode.NONE:
            msg = f"Unknown validation split mode: {self.val_split_mode}"
            raise ValueError(msg)
        
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Get training dataloader"""
        return DataLoader(
            dataset=self.train_data, 
            shuffle=True, 
            batch_size=self.train_batch_size,
            num_workers=self.num_workers, 
            collate_fn=self.train_data.collate_fn,
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Get validation dataloader."""
        return DataLoader(
            dataset=self.val_data, 
            shuffle=False, 
            batch_size=self.eval_batch_size, 
            num_workers=self.num_workers, 
            collate_fn=self.val_data.colate_fn,
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Get test dataloader."""
        return DataLoader(
            dataset=self.test_data,
            shuffle=False, 
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers, 
            collate_fn=self.test_data.collate_fn,
        )
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_dataloader
    
    @classmethod
    def from_config(
        cls: type["AnomalibDataModule"],
        config_path: str | Path, 
        **kwargs
    ) -> "AnomalibDataModule":
        """Create datamodule instance from config file
        
        Args: 
            config_path (str |Path): Path to config file
            **kwargs: Additional args to ovveride config

        Returns:
            AnomalibDataModule: Instantiated datamodule

        Example:
            Load from config file::

                >>> config_path = "examples/configs/data/mvtec.yaml"
                >>> datamodule = AnomalibDataModule.from_config(config_path)

            Override config values::

                >>> datamodule = AnomalibDataModule.from_config(
                ...     config_path,
                ...     data_train_batch_size=8
                ... )
        """
        from jsonargparse import ArgumentParser

        if not Path(config_path).exists():
            msg = f"Configuration file not found {config_path}"
            raise FileNotFoundError(msg)
        
        data_parser = ArgumentParser()
        data_parser.add_subclass_arguments(AnomalibDataModule, "data", required=False, fail_untyped=False)
        args = ["--data", str(config_path)]
        for key, value in kwargs.items():
            args.extend([f"--{key}", str(value)])
        config = data_parser.parse_args(args=args)
        instantiated_class = data_parser.instantiate_classes(config)
        datamodule = instantiated_class.get("data")
        if isinstance(datamodule, AnomalibDataModule):
            return datamodule 

        msg = f"Datamodule is not an instance of AnomalibDataModule: {datamodule}"
        raise ValueError(msg)
    