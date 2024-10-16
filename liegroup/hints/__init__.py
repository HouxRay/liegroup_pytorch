from typing import NamedTuple, Union

import torch
import numpy as np

# Type aliases for PyTorch/Numpy arrays; primarily for function inputs.

Array = Union[np.ndarray, torch.Tensor]
"""Type alias for `Union[torch.Tensor, np.ndarray]`."""

Scalar = Union[float, Array]
"""Type alias for `Union[float, Array]`."""


class RollPitchYaw(NamedTuple):
    """Tuple containing roll, pitch, and yaw Euler angles."""

    roll: Scalar
    pitch: Scalar
    yaw: Scalar


__all__ = [
    "Array",
    "Scalar",
    "RollPitchYaw",
]