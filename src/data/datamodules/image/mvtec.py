"""MVTec AD Data Module.

This module provides a PyTorch Lightning DataModule for the MVTec AD dataset. If
the dataset is not available locally, it will be downloaded and extracted
automatically.

Example:
        >>> from src.data import MVTec
        >>> datamodule = MVTec(
        ...     root="./datasets/mvtec",
        ...     category="bottle"
        ... )

Notes:
    The dataset will be automatically downloaded and converted to the required
    format when first used. The directory structure after preparation will be::

        datasets/
        └── mvtec/
            ├── bottle/
            ├── cable/
            └── ...
"""

import logging
from pathlib import Path 

from torchvision.transforms.v2 import Transform 

from src.data.datamodules.base.image import AnomalibDataModule
from src.data.datasets.image.mvtec import MVTecDataset 
from src.data.utils import DowloadInfo, Split, TestSplitMode, ValSplitMode, download_and_extract

logger = logging.getLogger(__name__)

DOWNLOAD_INFO = DownloadInfo(
    name="mvtec",
    url="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/"
    "download/420938113-1629952094/mvtec_anomaly_detection.tar.xz",
    hashsum="cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d",
)

class MVTec(AnomalibDataModule):
    """MVTec Datamodule
    
    root (Path | str): Path to the root of the dataset.
            Defaults to ``"./datasets/MVTec"``.
        category (str): Category of the MVTec dataset (e.g. ``"bottle"`` or
            ``"cable"``). Defaults to ``"bottle"``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        train_augmentations (Transform | None): Augmentations to apply dto the training images
            Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations to apply to the validation images.
            Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations to apply to the test images.
            Defaults to ``None``.
        augmentations (Transform | None): General augmentations to apply if stage-specific
            augmentations are not provided.
        test_split_mode (TestSplitMode): Method to create test set.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of data to use for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode): Method to create validation set.
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float): Fraction of data to use for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed for reproducibility.
            Defaults to ``None``.

    Example:
        Create MVTec datamodule with default settings::

            >>> datamodule = MVTec()
            >>> datamodule.setup()
            >>> i, data = next(enumerate(datamodule.train_dataloader()))
            >>> data.keys()
            dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

            >>> data["image"].shape
            torch.Size([32, 3, 256, 256])

        Change the category::

            >>> datamodule = MVTec(category="cable")

        Create validation set from test data::

            >>> datamodule = MVTec(
            ...     val_split_mode=ValSplitMode.FROM_TEST,
            ...     val_split_ratio=0.1
            ... )

    """
    def __init__(
        self, 
        root: Path | str = "./datasets/MVTec",
        category: str = "bottle", 
        train_batch_size: int = 32, 
        eval_batch_size: int = 32, 
        num_workers: int = 8, 
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augemtations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR, 
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.SAME_AS_TEST, 
        val_split_ratio: float = 0.5,
        seed: int | None = None, 
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size, 
            num_workers=num_workers, 
            train_augmentations=train_augmentations, 
            val_augmentations=val_augmentations, 
            augmentations=augmentations, 
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio, 
            val_split_mode=val_split_mode, 
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.root = Path(root)
        self.category = category
    
    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perfrom dynamic subset splitting
        
        This method may be overriden in subclass for custom splitting behavior.
        
        Note:
            The stage argument is not used here, as all three subsets are created in the
            same stage, at the first call of setup(). Due to anomaly detection tasks, 
            where val set is usally extracted frim test set. 
            Test set must therefore be created early as the `fit ` stage
        """
        self.train_data = MVTecDataset(
            split=Split.TRAIN, 
            root=self.root, 
            category=self.category,
        )
        self.test_data = MVTecDataset(
            split=Split.TEST,
            root=self.root, 
            category=self.category,
        )

    def prepare_data(self) -> None:
        """Download the dataset if not available. 
        
        This method checks if the specified dataset is available in the file
        system. If not, it downloads and extracts the dataset into 
        the appropriate directory.

        Example:
            Assume the dataset is not available in the filesystem
            >>> datamodule = MVTec(
            ...     root="./datasets/MVTec", 
            ...     category="bottle"
            ... )

            Directory structure after download
                datasets/
                └── MVTec/
                    ├── bottle/
                    ├── cable/
                    └── ...
        """
        if (self.root / self.category).is_dir()
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DOWNLOAD_INFO)
        
        