import math

import torch
from torch import nn
from torch.nn import functional as F

from efficient_ad_model_size import EfficientAdModelSize


def imagenet_norm_batch(x: torch.Tensor) -> torch.Tensor:
    """Normalize batch of images using ImageNet mean and standard deviation.

    This function normalizes a batch of images using the standard ImageNet mean and
    standard deviation values. The normalization is done channel-wise.

    Args:
        x (torch.Tensor): Input batch tensor of shape ``(N, C, H, W)`` where
            ``N`` is batch size, ``C`` is number of channels (3 for RGB),
            ``H`` is height and ``W`` is width.

    Returns:
        torch.Tensor: Normalized batch tensor with same shape as input, where each
            channel is normalized using ImageNet statistics:
            - Red channel: mean=0.485, std=0.229
            - Green channel: mean=0.456, std=0.224
            - Blue channel: mean=0.406, std=0.225
    """
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to(x.device)
    return (x - mean) / std

class SmallPatchDescriptionNetwork(nn.Module):
    """Small variant of Patch Description Network.
    
    Args: 
        out_channels (int): Number of output channels in the final convolution
        padding (bool, optional): Whether to use padding in the convolutional layers.
            Defaults to ``False``

    Example:
        >>> import torch
        >>> from src.efficientAD.efficient_ad_model import SmallPatchDescriptionNetwork
        >>> model = SmallPatchDescriptionNetwork(out_channels=384)
        >>> input_tensor = torch.randn(32, 3, 64, 64)
        >>> output = model(input_tensor)
        >>> output.shape
        torch.Size([32, 384, 13, 13])

    Notes:
        The network applies ImageNet normalization to the input before processing.
    """

    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv4 = nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor
        """Forward pass through the network. 

        Args: 
            x (torch.Tensor): Input tensor of shape ``(N, 3, H, W)``.

        Returns:
            torch.Tensor: Output features of shape
                ``(N, out_channels, H', W')`` where ``H'`` and ``W'`` are determined
                by the network architecture and padding settings
        """
        x = imagenet_norm_batch(x)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        return self.conv4(x)

class MediumPatchDescriptionNetwork(nn.Module):
    """Medium sized patch description network.
    
    Args: 
        out_channels (int): Number of output channels in the final convolutional layer
        padding (bool, optional): Whether to use padding in convolutional layers
            Defaults to ``false``
    Example:
        >>> import torch
        >>> from src.efficientAD.efficient_ad_model import MediumPatchDescriptionNetwork
        >>> model = MediumPatchDescriptionNetwork(out_channels=384)
        >>> input_tensor = torch.randn(32, 3, 64, 64)
        >>> output = model(input_tensor)
        >>> output.shape
        torch.Size([32, 384, 13, 13])
    Note: 
        The network applies ImageNet normalization to the input before processing.

    """
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0 * pad_mult)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv5 = nn.Conv2d(512, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.conv6 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0 * pad_mult)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args: 
            x (torch.Tensor): Input tensor of shape ``(N, 3, H, W)``.
        
        Returns:
            torch.Tensor: Output maps of shape
                ``(N, out_channels, H', W'``). Where ``H'`` and ``W'`` are determined 
                by the network architecture and padding settings.
        """
        x = imagenet_norm_batch(x)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return self.conv6(x)
    
class Encoder(nn.Module):
    """Encoder module for the autoencoder architecture
    
    The encoder consists of 6  convolutional layers that progressively reduce the spatial
    dimensions while increasing the number of channels

    Example:
        >>> import torch
        >>> from src.efficientAD.efficient_ad_model import Encoder
        >>> input_tensor = torch.randn(32, 3, 256, 256)
        >>> output = model(input_tensor)
        >>> output.shape
        torch.Size([32, 64, 1, 1])

    Note:
        The encoder uses ReLu activation after each convolutional layer 
        except the last one
    """
    def __init__(self) -> None:
        super().__init__()
        self.enconv1 = nn.Conv2d(3, 32, kernel_size=4, strid=2, padding=1)
        self.enconv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv6 = nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder network.

        Args:
            x (torch.Tensor): Input tensor of shape ``(N, 3, H, W)``.

        Returns:
            torch.Tensor: Encoded features of shape ``(N, 64, H', W')``, where
                ``H'`` and ``W'`` are determined by the network architecture.
        """
        x = F.relu(self.enconv1(x))
        x = F.relu(self.enconv2(x))
        x = F.relu(self.enconv3(x))
        x = F.relu(self.enconv4(x))
        x = F.relu(self.enconv5(x))
        return self.enconv6(x)

class Decoder(nn.Module):
    """Decoder module for the autoencoder architecture.

    The decoder consists of 8 convolutional layers with upsampling that
    progressively increase spatial dimensions while maintaining or reducing
    channel dimensions.

    Args:
        out_channels (int): Number of output channels in final conv layer.
        padding (int): Whether to use padding in convolutional layers.

    Example:
        >>> import torch
        >>> from src.efficientAD.efficient_ad_model import Decoder
        >>> model = Decoder(out_channels=384, padding=True)
        >>> input_tensor = torch.randn(32, 64, 1, 1)
        >>> image_size = (256, 256)
        >>> output = model(input_tensor, image_size)
        >>> output.shape
        torch.Size([32, 384, 64, 64])

    Note:
        - Uses ReLU activation and dropout after most convolutional layers
        - Performs bilinear upsampling between conv layers to increase spatial
          dimensions
        - Final output size depends on ``padding`` parameter and input
          ``image_size``
    """

    def __init__(self, out_channels: int, padding: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.padding = padding
        # use ceil to match output shape of PDN
        self.deconv1 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv6 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.2)
        self.dropout6 = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor, image_size: tuple[int, int] | torch.Size) -> torch.Tensor:
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape ``(N, 64, H, W)``.
            image_size (tuple[int, int] | torch.Size): Target output size
                ``(H, W)``.

        Returns:
            torch.Tensor: Decoded features of shape
                ``(N, out_channels, H', W')``, where ``H'`` and ``W'`` are
                determined by the network architecture and padding settings.
        """
        last_upsample = (
            math.ceil(image_size[0] / 4) if self.padding else math.ceil(image_size[0] / 4) - 8,
            math.ceil(image_size[1] / 4) if self.padding else math.ceil(image_size[1] / 4) - 8,
        )
        x = F.interpolate(x, size=(image_size[0] // 64 - 1, image_size[1] // 64 - 1), mode="bilinear")
        x = F.relu(self.deconv1(x))
        x = self.dropout1(x)
        x = F.interpolate(x, size=(image_size[0] // 32, image_size[1] // 32), mode="bilinear")
        x = F.relu(self.deconv2(x))
        x = self.dropout2(x)
        x = F.interpolate(x, size=(image_size[0] // 16 - 1, image_size[1] // 16 - 1), mode="bilinear")
        x = F.relu(self.deconv3(x))
        x = self.dropout3(x)
        x = F.interpolate(x, size=(image_size[0] // 8, image_size[1] // 8), mode="bilinear")
        x = F.relu(self.deconv4(x))
        x = self.dropout4(x)
        x = F.interpolate(x, size=(image_size[0] // 4 - 1, image_size[1] // 4 - 1), mode="bilinear")
        x = F.relu(self.deconv5(x))
        x = self.dropout5(x)
        x = F.interpolate(x, size=(image_size[0] // 2 - 1, image_size[1] // 2 - 1), mode="bilinear")
        x = F.relu(self.deconv6(x))
        x = self.dropout6(x)
        x = F.interpolate(x, size=last_upsample, mode="bilinear")
        x = F.relu(self.deconv7(x))
        return self.deconv8(x)
    
class AutoEncoder(nn.Module):
    """EfficientAD Autoencoder.
    
    The autoencoder consists of an encoder and decoder network. The encoder extracts features
    from the input image which are the reconstructed by the decoder

    Args: 
        out_channels (int): Number of convolution output channels in the decoder.
        padding (int) Whether to ised padding in the convolutinal layers.
        *args: Variable length argument list pass to parent class
        **kwargs: Arbitrary keyword arguments passt to parent class

    Example:
        >>> from torch import randn
        >>> autoencoder = AutoEncoder(out_channels=384, padding=True)
        >>> input_tensor = randn(32, 3, 256, 256)
        >>> output = autoencoder(input_tensor, image_size([256, 256]))
        >>> output.shape
        torch.Size([32, 384, 256, 256])
    
    Note: 
        The input images are normalized using ImageNet statistics before beeing
        passed through the encoder. 
    
    """

    def __init__(self, out_channels: int, padding: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder: Encoder = Encoder()
        self.decoder: Decoder = Decoder(out_channels=out_channels, padding=padding)

    def forward(self, x: torch.Tensor, image_size: tuple[int, int] | torch.Size) -> torch.Tensor:
        """Forward pass through autoencoder
        
        Args: 
            x (torch.Tensor): Input tensor of shape ``(N, C, H, W)``
            image_size (tuple[int, int] | torch.Size): Targe output size ``(H, W)``

        Returns 
            torch.Tensor: Reconstructed features of shape ``(N, out_channels, H', W')
            where ``H'`` and ``W'`` are determined by the architecture and padding settings        
        """
        x = imagenet_norm_batch(x)
        x = self.encoder(x)
        return self.decoder(x, image_size)
    
class EfficientAdModel(nn.Module):
    """EfficientAd model.
    
    Consisting of teacher and student network for anomaly detection.
    The teacher network is pretrained and frozen.
    The student network is trained to match the teacher's output.

    Args:
        teacher_output_channels (int): Number of covolution output channels of the pre-trained teacher model
        model_size (EfficientAdModelSize): size of student and teacher model.
            Defaults to ``EfficentAdModelSize.S``
        padding (bool): Whter to use padding in convolutional layers.
            Defaults to ``False``
        pad_maps (bool): Wether to pad output anomaly maps when ``padding=False`` to match size of padded case
            Only relevant if ``padding=False``
            Defaults to ``True``

    Example:
        >>> from src.efficientAD.efficient_ad_model import EfficientAdModel
        >>> from src.eccicientAD.efficient_ad_model_size import EfficientAdModelSize
        >>> model = efficientAdModel(
                teacher_out_channels=384, 
                model_size=EfficientAdModelSize.S
            )
        >>> input tensor = torch.randn(32, 3, 256, 256)
        >>> output = model(input_tensor)
        >>> output.anomaly_map.shape
        torch.size([32, 1, 256, 256])
    
    
    
    """

    def __init__(
            self, 
            teacher_out_channels: int,
            model_size: EfficientAdModelSize = EfficientAdModelSize.S,
            padding: bool = False,
            pad_maps: bool = True,
            ) -> None:
        super().__init__()
        
        self.pad_maps = pad_maps
        self.teacher: MediumPatchDescriptionNetwork | SmallPatchDescriptionNetwork
        self.student: MediumPatchDescriptionNetwork | SmallPatchDescriptionNetwork

        if model_size == EfficientAdModelSize.M:
            self.teacher = MediumPatchDescriptionNetwork(out_channels=teacher_out_channels, padding=padding).eval()
            self.student = MediumPatchDescriptionNetwork(out_channels=teacher_out_channels * 2, padding=padding)

        elif model_size == EfficientAdModelSize.S:
            self.teacher = SmallPatchDescriptionNetwork(out_channels=teacher_out_channels, padding=padding).eval()
            self.student = SmallPatchDescriptionNetwork(out_channels=teacher_out_channels * 2, padding=padding)

        else:
            msg = f"Unknown model size {model_size}"
            raise ValueError(msg)
        
        self.ae: AutoEncoder = AutoEncoder(out_channels=teacher_out_channels, padding=padding)
        self.teacher_out_channels: int = teacher_out_channels

        self.mean_std: nn.ParamDict = nn.ParameterDict (
            {
                "mean": torch.zeros((1, self.teacher_out_channels, 1, 1)),
                "std": torch.zeros((1, self.teacher_out_channels, 1, 1)),
            }
        )

        self.quantiles: nn.ParameterDict = nn.ParameterDict(
            {
                "qa_st": torch.tensor(0.0),
                "qb_st": torch.tensor(0.0),
                "qa_ae": torch.tensor(0.0),
                "qb_ae": torch.tensor(0.0),
            }
        )

    @staticmethod
    def is_set(p_dic: nn.ParameterDict) -> bool:
        pass 

    @staticmethod
    def choose_random_aug_imae(image: torch.Tensor) -> torch.Tensor:
        pass 

    def forward(
        self,
        batch: torch.Tensor,
        batch_imagenet: torch.Tensor | None = None,
        normalize: bool = True, 
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | InferenceBatch:
        pass 

    def compute_student_and_teacher_distance(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass
    
    def compute_losses(
        self,
        batch: torch.Tensor, 
        batch_imagenet: torch.Tensor, 
        distacne_st: torch.Tensor, 
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass 

    def compute_maps(
        self,
        batch: torch.Tensor, 
        student_output: torch.Tensor, 
        distance_st: torch.Tensor,
        normalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass 

    def get_maps(self, batch: torch.Tensor, normalize: bool = False) -> tuple[torch.Tensor, torch.Tensor]: 
        pass

