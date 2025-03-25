"""Cutom exceptions for anomalit data validation

This module provides custom exception classes for handling data validations 
in anomalib
"""

class MisMatchError(Exception):
    """Exception raised whrn a dataset mismatch is detected
    
    This exception is raised when there is a mismatch between 
    expeceted and actual data formats or values during validation.

    Args:
        message (str): Custom error message. 
            Defaukts to ``"mismatch detected"``

    Attributes:
        messgae (str): Explanation of the error.

    Example:
        >>> raise MisMatchError # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        MisMatchError: Mismatch detected.
        >>> raise MisMatchError("Image dimensions do not match")
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        MisMatchError: Image dimensions do not match
    """

    def __init__(self, message: str = "") -> None:
        if message:
            self.message = message
        else:
            self.message = "Mismatch detected."
        super().__init__(self.message)