"""EfficientAD: Accurate visual Anomaly Detection

The model uses a student-teacher approach with a pre-trained EfficientNet backbone

The model consists of:
    - A pre-trained EfficientNet teacher network
    - A lightweight student network
    - Knowledge destillation traing
    - Anomaly detection via feature comparison

"""
from pathlib import Path

from efficientAD.efficient_ad_model_size import EfficientAdModelSize 
from efficientAD.efficient_ad_model import EfficientADModel

from visuaization import Visualizer
from pre_processing import PreProcessor
from post_processing import PostProcessor
from evaluation import Evaluator


print("Hello World")


class EfficientAD():
    """PL Module for EfficientAD algorithm
        
    Args:
        imagenet_dir (Path | str): Directory path for Imagenet dataset.
            Defaults to ``"./datasets/imagenette"`` 
        teacher_out_channels (int): Number of convolution output channels.
            Defaults to ``384``
        model_size (EfficientAdModelSize | str): Size of student and teacher model
            Defaults to ``EfficientAdModelSize.S``
        lr (float):learning rate
            Defaults to ``0.0001``
        weight_decay (float): Optimizer weigth decay
            Defaults to ``0.00001``
        padding (bool): Use padding in convoltional layers.
            Defaults to ``False``
        pad_maps (bool): Relevant if ``padding=False``. If ``True``, pads the anomaly maps to match the size of ``padding=True`` case 
            Defaults to ``True``
        pre_processor (PreProcessor | bool, optional): Preprocessor used to transform input data before passing to model.
            Defaults to ``True``
        post_processor (PostProcessor | bool, optional): Post-processor used to prcess model predictions
            Defaults to ``True``
        evaluator (Evaluator | bool, optional): Evaluator used to compute metrics
            Defaults to ``True``
        visualizer (Visualizer| bool, optional): Visualizer used to create visualizations
            Defaults to ``True``
            
    """

    def __init__(
            self, 
            imagenet_dir: Path | str = "./datasets/imagenette",
            teacher_out_channels: int = 384, 
            model_size: EfficientAdModelSize | str = EfficientAdModelSize.S, 
            lr: float = 0.0001,
            weight_decay: float = 0.00001,
            padding: bool = False,
            pad_maps: bool = True,
            pre_processor: PreProcessor | bool = True,
            post_processor: PostProcessor | bool = True,
            evaluator: Evaluator | bool = True,
            visualizer: Visualizer | bool = True,
    ) -> None:
        self.imagenet_dir = imagenet_dir
        if not isinstance(model_size, EfficientAdModelSize):
            model_size = EfficientAdModelSize(model_size)
        self.model_size: EfficientAdModelSize = model_size
        self.model: EfficientADModel(
            teacher_out_channels=teacher_out_channels, 
            model_size=model_size, 
            padding=padding, 
            pad_maps=pad_maps,
        )
        self.batch_size: int = 1 # imagenet dataloader batch_size is 1
        self.lr: float = lr
        self.weight_decay: float = weight_decay

    def prepare_pretrained_model(self) -> None:
        pass 

    def prepare_imagenette_data(self, image_size: tuple[int, int] | torch.Size) -> None:
        pass 

    @torch.no_grad
    def teacher_channel_mean_std(self, dataloader: Dataloader) -> dict[str, torch.Tensor]:
        pass 



    
    
