from typing import TYPE_CHECKING, Callable, Tuple, Type, TypeVar, Union, cast

import torch
from torch import nn

from ..hints import Array

if TYPE_CHECKING:
    from _base import MatrixLieGroup


T = TypeVar("T", bound="MatrixLieGroup")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_epsilon(dtype: torch.dtype) -> float:
    """Helper for grabbing type-specific precision constants.

    Args:
        dtype: Datatype.

    Returns:
        Output float.
    """
    return {
        torch.float32: 1e-5,
        torch.float64: 1e-10,
    }[dtype]


def register_lie_group(
    *,
    matrix_dim: int,
    parameters_dim: int,
    tangent_dim: int,
    space_dim: int,
) -> Callable[[Type[T]], Type[T]]:
    """Decorator for registering Lie group classes.

    Sets dimensionality class variables.
    """

    def _wrap(cls: Type[T]) -> Type[T]:
        # Register dimensions as class attributes.
        cls.matrix_dim = matrix_dim
        cls.parameters_dim = parameters_dim
        cls.tangent_dim = tangent_dim
        cls.space_dim = space_dim

        return cls

    return _wrap


TupleOfBroadcastable = TypeVar(
    "TupleOfBroadcastable",
    bound="Tuple[Union[MatrixLieGroup, Array], ...]",
)


def broadcast_leading_axes(inputs: TupleOfBroadcastable) -> TupleOfBroadcastable:
    """Broadcast leading axes of arrays. Takes tuples of either:
    - a tensor, which we assume has shape (*, D).
    - a Lie group object."""

    from .._base import MatrixLieGroup

    tensor_inputs = [
        (
            (x.parameters(), (x.parameters_dim,))
            if isinstance(x, MatrixLieGroup)
            else (x, x.shape[-1:])
        )
        for x in inputs
    ]
    for tensor, shape_suffix in tensor_inputs:
        assert tensor.shape[-len(shape_suffix):] == shape_suffix
    batch_axes = torch.broadcast_shapes(
        *[tensor.shape[:-len(suffix)] for tensor, suffix in tensor_inputs]
    )
    broadcasted_tensors = tuple(
        torch.broadcast_to(tensor, batch_axes + shape_suffix).to(device)
        for (tensor, shape_suffix) in tensor_inputs
    )
    return cast(
        TupleOfBroadcastable,
        tuple(
            tensor if not isinstance(inp, MatrixLieGroup) else type(inp)(tensor)
            for tensor, inp in zip(broadcasted_tensors, inputs)
        ),
    )