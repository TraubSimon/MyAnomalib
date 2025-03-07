"""Generic dataclasses that can be implemented for different datatypes.

This module provides a set of generic dataclasses and mixins that can be used
to define and validate various types of data fileds used in Anomalib. 
The dataclasses are designed to be flexible and extensible, allowing for easy
customization and validation of input and output data.

The module cotains several key components:
- Filed descriptors for validation
- Base input field classes for images 
- Output filed classes for predictions
- Mixins for updating and batch iteration
- Generic item and batch classes

Example:
    >>> from src.data.dataclasses import _InputFileds
    >>> from torchvision.tv_tensors import Image, Mask
    >>>
    >>> class MyInput(_InputFileds[int, Image, Mask, str]):
    ...     def validate_image(self, image):
    ...         return image
    ...     # implement other valudation methods
    ...
    >>> input_data = MyInput(
    ...     image=torch.rand((3, 224, 224)),
    ...     gt_label=1, 
    ...     gt_mask=None, 
    ...     mask_path=None
    ... )
"""

from abc import ABC 
from typing import Generic, TypeVar, get_args, get_type_hints
from types import NoneType

PathT = TypeVar("PathT", list[str], str)
Value = TypeVar("Value")
Instance = TypeVar("instance")

class FieldDescriptor(Generic[Value]):
    """Desciptor for Anomalib's dataclass fields.
    
    Using a descriptor ensures that the value of a dataclass fields can be 
    validated before being set. This allows validation of the input data not
    only when it is first set, but also when it is updated.

    Args:
        validator_name: Name of the validator mathod to call when setting value.
            Defaults to `None`
        default: Default value for the filed
            Defaults to `None`

    Example:
        >>> class MyClass:
        ...     field = FiledDescriptor(validator_name="validate_field")
        ...     def validata_field(self, value):
        ...         return value
        ...
        >>> obj = MyClass()
        >>> obj.field = 42
        >>> obj.field
        42
    """

    def __init__(self, validator_name: str | None = None, default: Value | None = None) -> None:
        """Initialize the descriptor."""
        self.validator_name =validator_name
        self.default = default

    def __set_name__(self, owner: type[Instance], name: str) -> None:
        """Set the name of the descriptor.
        
        Args:
            owner: Class that owns the descriptor
            name: Name of the descriptor
        """
        self.name = name 

    def __get__(self, instance: Instance | None, owner: type[Instance]) -> Value | None:
        """Get the values of the descriptor.
        
        Args:
            instance: INstance the descriptor is accessed from
            owner: Class that owns the descriptor
            
        Returns: 
            Default value if instance is None, otherwise the stored value

        Raises:
            AttributeError: If no default value and filed is not optional            
        """
        if instance is None:
            if self.default is not None or self.is_optional(owner):
                return self.default 
            msg = f"No default attribute value specified for field '{self.name}'."
            raise AttributeError(msg)
        return instance.__dict__[self.name]
    

    def __set__(self, instance: object, value: Value) -> None:
        """Set the value of the descriptor.
        
        First calls the validator method if available, then sets the value.
        
        Args: 
            instance: Instance to set the Value on
            value: Value to set
        """
        if self.validator_name is not None:
            validator = getattr(instance, self.validator_name)
            value = validator(value)
        instance.__dict__[self.name] = value 

    def __get_types__(self, owner: type[Instance]) -> tuple[type, ...]:
        """Get the types of the descriptor
        
        Args: 
            owner: Class that owns the descriptor
            
        Returns: 
            Tuple of valid types for this field.

        Raises: 
            TypeError: If types cannot be determined                      
        """
        try:
            types = get_args(get_type_hints(owner)[self.name])
            return get_args(types[0]) if hasattr(types[0], "__args__") else (types[0], )
        except (KeyError, TypeError, AttributeError) as e:
            msg = f"Unable to detemine types for {self.name} in {owner}"
            raise TypeError(msg) from e 
        
    def is_optional(self, owner: type[Instance]) -> bool:
        """Check if the descriptor is optional.
        
        Args: 
            owner: Class that owns the descriptor
            
        Returns: 
            True if field can be None, False otherwise
        """
        return NoneType in self.get_types(owner)
    

class _ImageInputFields(Generic[PathT], ABC):
    """Generic dataclass for image specific input fileds
    
    This class estends standar input fileds with an `image_path` attribute
    for image-based anomaly detection tasks.

    Attributes:
        image_path: Path to input image file

    Example:
        >>> class MyImageInput(_ImageInputFileds[str]):
        ...     def validate_image_path(self, path):
        ...         return path
        ...
        >>> input_data = MyImageInput(image_path="path/to/image.jpg")   
    """

    image_path: FieldDescriptor[PathT | None] = FieldDescriptor(validator_name="validate_image_path")
    
    def __init__(self):
        
























class BatchIteratorMixin: 
    def __init__(self):
         pass 
        
