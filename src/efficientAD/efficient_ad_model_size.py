from enum import Enum

class EfficientAdModelSize(str, Enum):
    """Supported EfficientAd model sizes. 
    
    The EfficientAd comes in two model sizes: 
        - ``M``` (medium): Uses larger architecture with more parameters
        - ``S`` (small): Uses a smaller architecture with fewer parameters
    
    Example:
        >>> from EfficientAdModelSize import EfficientAdModelSize
        
        >>> model_size = EfficientAdModelSize.S
        >>> model_size
        'small'
        
        >>> model_size = EffiecientAdModelSize.M
        >>> model_size
        'medium'
    
    """

    M = "medium"
    S = "small"