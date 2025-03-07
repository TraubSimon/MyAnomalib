"""Multi random choice transform

This transform applies muultiple tranforms form a list of transforms

Example: 
    >>> import torchvision.transform.v2 as v2
    >>> transforms = [
    ...     v2.RandomHorizontalFlip(p=1.0),
    ...     v2.ColorJitter(brightness=0.5), 
    ...     v2.RandomRotation(10),
    ... ]
    >>> # Apply 1-2 random transforms with equal probability
    >>> transforms = MultiRandomChoice(transform, num_transforms=2)
    >>> # Always apply exactly 2 transfomrs with custom probabilities
    >>> transform = MultiRandomChoice(
    ...     transforms, 
    ...     probabilities=[0.5, 0.3, 0.2], 
    ...     num_transforms=2, 
    ...     fixed_num_transforms=True
    ... )
"""


from collections.abc import Callable, Sequence

import torch
from torchvision.transforms import v2 

class MultiRandomChoice(v2.Transform):
    """Apply multiple trnasforms randomly picked from a list
    
    This transform does not support torchscript.
    
    Args:
        transforms (List[Transforms]): List of transforms to choose from
        probabilities (List[float] | None): Probability of each transform being picked. If `None`
            (default), all transforms have equal probability. If provided, probabilities will be 
            normalized to sum to 1.
        num_transforms (int): Maximum number of transforms to apply at once.
            Defaults to `1`
        fixed_num_tranforms (bool): If `True`, always applies exactly `num_trnasfomrs` transforms. If `False`
            randomly picks between 1 and `num_transfomrs`.
            Defaults to `False`

    Raises:
        TypeError: If `transforms` is not a swquence of callables
        ValueError: If length of `probabilities` does not match the length
            of `transfomrs`.

    Example:
        >>> import torchvision.transforms.v2 as v2
        >>> transforms = [
        ...     v2.RandomHorizontalFlip(p=1.0),
        ...     v2.ColorJitter(brightness=0.5),
        ...     v2.RandomRotation(10),
        ... ]
        >>> # Apply 1-2 random transforms with equal probability
        >>> transform = MultiRandomChoice(transforms, num_transforms=2)
        >>> # Always apply exactly 2 transforms with custom probabilities
        >>> transform = MultiRandomChoice(
        ...     transforms,
        ...     probabilities=[0.5, 0.3, 0.2],
        ...     num_transforms=2,
        ...     fixed_num_transforms=True
        ... )
    """

    def __init__(
        self, 
        transforms: Sequence[Callable], 
        probabilities: list[float] | None = None,
        num_transfomrs: int = 1,
        fixed_num_transforms: bool = False,
    ) -> None:
        if not isinstance(transforms, Sequence):
            msg = "Argument transform should be a sequence of callables"
            raise TypeError(msg)
        elif len(transforms) != len(probabilities):
            msg = f"Length of transforms and probabilities do not match: {len(probabilities)} != {len(transforms)}"
            raise ValueError(msg)
        
        super().__init__()

        self.transforms = transforms
        total = sum(probabilities)
        self.probabilites = [probability / total for probability in probabilities]

        self.num_transforms = num_transfomrs
        self.fixed_num_transforms = fixed_num_transforms

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Apply randomly selected transforms to the input."""
        # First determine the number of transforms to appply
        num_transforms = (
            self.num_transforms if self.fixed_num_transfomrs else int(torch.randint(self.num_transforms, (1,)) + 1)
        )
        # Get transforms
        idx = torch.multinomial(torch.tensor(self.probabilites), num_transforms).tolist()
        transform = v2.Compose([self.transform[i] for i in idx])
        return transform
        