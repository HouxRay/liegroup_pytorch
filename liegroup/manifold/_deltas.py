"""Helpers for recursively applying tangent-space deltas."""

from typing import Any, Callable, TypeVar, Union, cast, overload

import torch

from .. import hints
from .._base import MatrixLieGroup
from .._se2 import SE2
from .._se3 import SE3
from .._so2 import SO2
from .._so3 import SO3
from . import _tree_utils

device = "cuda" if torch.cuda.is_available() else 'cpu'

PytreeType = TypeVar("PytreeType")
GroupType = TypeVar("GroupType", bound=MatrixLieGroup)

def _rplus(transform: GroupType, delta: torch.Tensor) -> GroupType:
    assert isinstance(transform, MatrixLieGroup)
    assert isinstance(delta, torch.Tensor)
    return transform @ type(transform).exp(delta)


@overload
def rplus(
    transform: GroupType,
    delta: hints.Array,
) -> GroupType: ...

# @overload
# def rplus(
#     transform: PytreeType,
#     delta: _tree_utils.TangentPytree,
# ) -> PytreeType: ...


# Using our typevars in the overloaded signature will cause errors.
def rplus(
    transform: Union[MatrixLieGroup, Any],
    delta: Union[hints.Array, Any],
) -> Union[MatrixLieGroup, Any]:
    """Manifold right plus. Computes `T' = T @ exp(delta)`.

    Supports pytrees containing Lie group instances recursively; simple Euclidean
    addition will be performed for all other arrays.
    """
    return _tree_utils._map_group_trees(_rplus, torch.add, transform, delta)


@overload
def rminus(a: GroupType, b: GroupType) -> torch.Tensor: ...

def _rminus(a: GroupType, b: GroupType) -> torch.Tensor:
    assert isinstance(a, MatrixLieGroup) and isinstance(b, MatrixLieGroup)
    return (a.inverse() @ b).log()


# Using our typevars in the overloaded signature will cause errors.
def rminus(
    a: Union[MatrixLieGroup, Any], b: Union[MatrixLieGroup, Any]
) -> Union[torch.Tensor]:
    """Manifold right minus. Computes
    `delta = T_ab.log() = (T_wa.inverse() @ T_wb).log()`.

    Supports pytrees containing Lie group instances recursively; simple Euclidean
    subtraction will be performed for all other arrays.
    """
    return _tree_utils._map_group_trees(_rminus, torch.subtract, a, b)


# @torch.jit.script
def rplus_jacobian_parameters_wrt_delta(transform: MatrixLieGroup) -> torch.Tensor:
    """Analytical Jacobians for `jaxlie.manifold.rplus()`, linearized around a zero
    local delta.

    Mostly useful for reducing JIT compile times for tangent-space optimization.

    Equivalent to --
    ```
    def rplus_jacobian_parameters_wrt_delta(transform: MatrixLieGroup) -> jax.Array:
        # Since transform objects are pytree containers, note that `jacfwd` returns a
        # transformation object itself and that the Jacobian terms corresponding to the
        # parameters are grabbed explicitly.
        return jax.jacfwd(
            jaxlie.manifold.rplus,  # Args are (transform, delta)
            argnums=1,  # Jacobian wrt delta
        )(transform, onp.zeros(transform.tangent_dim)).parameters()
    ```

    Args:
        transform: Transform to linearize around.

    Returns:
        Jacobian. Shape should be `(Group.parameters_dim, Group.tangent_dim)`.
    """
    if isinstance(transform, SO2):
        # Jacobian row indices: cos, sin
        # Jacobian col indices: theta

        J = torch.zeros((*transform.get_batch_axes(), 2, 1)).to(device)
        cos, sin = torch.moveaxis(transform.unit_complex, -1, 0).to(device)
        J[..., 0, 0] = -sin
        J[..., 1, 0] = cos

    elif isinstance(transform, SE2):
        # Jacobian row indices: cos, sin, x, y
        # Jacobian col indices: vx, vy, omega
        J = torch.zeros((*transform.get_batch_axes(), 4, 3)).to(device)

        # Translation terms.
        J[..., 2:, :2] = transform.rotation().as_matrix()

        # Rotation terms.
        J[..., :2, 2:3] = rplus_jacobian_parameters_wrt_delta(transform.rotation())
        

    elif isinstance(transform, SO3):
        # Jacobian row indices: qw, qx, qy, qz
        # Jacobian col indices: omega x, omega y, omega z
        w, x, y, z = torch.moveaxis(transform.wxyz, -1, 0).to(device)
        neg_x = -x
        neg_y = -y
        neg_z = -z

        J = (
            torch.stack(
                [
                    neg_x,
                    neg_y,
                    neg_z,
                    w,
                    neg_z,
                    y,
                    z,
                    w,
                    neg_x,
                    neg_y,
                    x,
                    w,
                ],
                axis=-1,
            ).reshape((*transform.get_batch_axes(), 4, 3))
            / 2.0
        )

    elif isinstance(transform, SE3):
        # Jacobian row indices: qw, qx, qy, qz, x, y, z
        # Jacobian col indices: vx, vy, vz, omega x, omega y, omega z
        J = torch.zeros((*transform.get_batch_axes(), 7, 6)).to(device)

        # Translation terms.
        J[..., 4:, :3] = (transform.rotation().as_matrix())

        # Rotation terms.
        J[..., :4, 3:6] = rplus_jacobian_parameters_wrt_delta(transform.rotation())
        

    else:
        assert False, f"Unsupported type: {type(transform)}"

    assert J.shape == (
        *transform.get_batch_axes(),
        transform.parameters_dim,
        transform.tangent_dim,
    )
    return J