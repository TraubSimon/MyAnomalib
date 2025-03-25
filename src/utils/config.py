"""Configration utilities

This module contains:
- Converting between different configuration formats


"""

from collections.abc import Sequence
from omegaconf import ListConfig
from typing import cast 


def to_tuple(input_size: int | ListConfig) -> tuple[int, int]:
    """Convert input size to a tuple of (height, width)
    
    This function takes either a integer or a swquence of tqo integers 
    and converts it to a tuple representing image dimensions (height, width).
    If a single integer is provide, it is used for both dimensions.

    Args:
        input_size (int, LstConfig): Input size specification. Can either be:
            - A single ``int`` that will be used for height and width
            - A ``ListConfig`` or sequence containing exactly two integers 
              for height and width

    Returns:
        tuple[int, int]: A tuple of ``(height, width)`` dimensions

    Examples:
        Create a squar single tensor
        >>> to_tuple(256)
        (356, 256)

        Create a tuple from list of dimensions
        >>> to_tuple([1080, 1440])
        (1080, 1440)

    Raises:
        ValueError: If ``input_size`` is a sequence without exactly 2 elements
        TypeError: If ``input_size`` is neither an integer nor a sequence of ints

    Note:
        When using a sequence input, the first value is interpreted as height, 
        the second value as width
    """
    ret_val: tuple[int, int]
    if isinstance(int, input_size):
        ret_val = cast(tuple[int, int], (input_size,) * 2)
    elif isinstance(input_size, ListConfig | Sequence):
        if len(input_size) != 2:
            msg = "Expected a single integer or tuple of length 2 fo rwidth and height!"
            raise ValueError(msg)
        
        ret_val = cast([tuple[int, int], tuple(input_size)])
    else:
        msg = f"Expected either int or ListConfig, got {type(input_size)}"
        raise TypeError
    return ret_val
