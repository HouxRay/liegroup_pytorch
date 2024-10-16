"""Test manifold helpers."""

from typing import Tuple, Type

import torch
import functorch
from torch.autograd.functional import jacobian
import numpy as onp
import pytest
from tests.liegroup_test_utils import (
    assert_arrays_close,
    assert_transforms_close,
    general_group_test,
    general_group_test_faster,
    sample_transform,
)
import liegroup.manifold
import liegroup


device = "cuda" if torch.cuda.is_available() else 'cpu'

@general_group_test
def test_rplus_rminus(Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Check rplus and rminus on random inputs."""
    T_wa = sample_transform(Group, batch_axes)
    T_wb = sample_transform(Group, batch_axes)
    T_ab = T_wa.inverse() @ T_wb

    assert_transforms_close(liegroup.manifold.rplus(T_wa, T_ab.log()), T_wb)
    assert_arrays_close(liegroup.manifold.rminus(T_wa, T_wb), T_ab.log())


@general_group_test
def test_rplus_jacobian(
    Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    """Check analytical rplus Jacobian.."""
    T_wa = sample_transform(Group, batch_axes)

    J_ours = liegroup.manifold.rplus_jacobian_parameters_wrt_delta(T_wa)

    if batch_axes == ():
        J_jacfwd = _rplus_jacobian_parameters_wrt_delta(T_wa)
        assert_arrays_close(J_ours, J_jacfwd)
    else:
        # Batch axes should match vmap.
        jacfunc = liegroup.manifold.rplus_jacobian_parameters_wrt_delta
        for _ in batch_axes:
            jacfunc = functorch.vmap(jacfunc)
        J_vmap = jacfunc(T_wa)
        assert_arrays_close(J_ours, J_vmap)


# @torch.jit.script
def _rplus_jacobian_parameters_wrt_delta(
    transform: liegroup.MatrixLieGroup,
) -> torch.Tensor:
    # Copied from docstring for `rplus_jacobian_parameters_wrt_delta()`.
    # return torch.jacfwd(
    #     lambda delta: liegroup.manifold.rplus(transform, delta).parameters()
    # )(torch.zeros(transform.tangent_dim))
    return jacobian(lambda delta: liegroup.manifold.rplus(transform, delta).parameters(), 
                    torch.zeros(transform.tangent_dim).to(device))


@general_group_test_faster
def test_sgd(Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    def loss(transform: liegroup.MatrixLieGroup):
        return (transform.log() ** 2).sum()

    transform = Group.exp(sample_transform(Group, batch_axes).log())
    original_loss = loss(transform)

    # @torch.jit.script
    def step(t):
        return liegroup.manifold.rplus(t, -1e-3 * liegroup.manifold.grad(loss)(t))

    for i in range(5):
        transform = step(transform)

    assert loss(transform) < original_loss

def test_rplus_euclidean():
    assert_arrays_close(
        liegroup.manifold.rplus(torch.ones(2), torch.ones(2)), 2 * torch.ones(2)
    )

    
def tree_map(func, tree):
    """Recursively apply `func` to each leaf in `tree`."""
    if isinstance(tree, (list, tuple)):
        return type(tree)(tree_map(func, item) for item in tree)
    elif isinstance(tree, dict):
        return {key: tree_map(func, value) for key, value in tree.items()}
    else:
        return func(tree)


def test_rminus_auto_vmap():
    torch.manual_seed(0)
    sample_uniform = liegroup.SE3.sample_uniform(round(torch.randn(1).item()))
    identity = liegroup.SE3.identity()

    args = [
        sample_uniform,
        identity,
    ]
    args_rev = [
        identity,
        sample_uniform,
    ]       

    deltas = liegroup.manifold.rminus(
        tree_map(
            lambda *se3s: torch.stack([se3.wxyz_xyz for se3 in se3s]),  # 提取 wxyz_xyz
            args,
        ),
        tree_map(
            lambda *se3s: torch.stack([se3.wxyz_xyz for se3 in se3s]),
            args_rev,
        ),
    )
    assert_arrays_close(deltas[0], -deltas[1])
    

def test_normalize():
    container = {"key": (liegroup.SO3(torch.tensor([2.0, 0.0, 0.0, 0.0])),)}
    container_valid = {"key": (liegroup.SO3(torch.tensor([1.0, 0.0, 0.0, 0.0])),)}
    with pytest.raises(AssertionError):
        assert_transforms_close(container["key"][0], container_valid["key"][0])
    assert_transforms_close(
        liegroup.manifold.normalize_all(container)["key"][0], container_valid["key"][0]
    )


if __name__ == "__main__":
    test_rplus_euclidean()
    print("test_rplus_euclidean passed")
    test_rplus_rminus(liegroup.SE3, batch_axes=())
    print("test_rplus_rminus passed")
    test_rplus_jacobian(liegroup.SE2, batch_axes=())
    print("test_rplus_jacobian passed")
    test_normalize()
    print("test_normalize passed")
    test_rminus_auto_vmap()
    print("test_rminus_auto_vmap passed")
    test_sgd(liegroup.SO2, batch_axes=())
    print("test_sgd passed")