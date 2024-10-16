"""Compare forward- and reverse-mode Jacobians with a numerical Jacobian."""
import functorch

import liegroup

from functools import lru_cache
from typing import Callable, Tuple, Type

import torch
from torch.autograd.functional import jacobian
from tests.liegroup_test_utils import assert_arrays_close, general_group_test, jacnumerical


device = "cuda" if torch.cuda.is_available() else 'cpu'

cached_jacfwd = lru_cache(maxsize=None)(
    lambda f: torch.jit.trace(functorch.jacfwd(f),torch.tensor([0.1]))
)
cached_jacrev = lru_cache(maxsize=None)(
    lambda f: torch.jit.trace(functorch.jacrev(f),torch.tensor([0.1]))
)
cached_jit = lru_cache(maxsize=None)(torch.jit.trace)


def _assert_jacobians_close(
    Group: Type[liegroup.MatrixLieGroup],
    f: Callable[[Type[liegroup.MatrixLieGroup], torch.Tensor], torch.Tensor],
    primal: liegroup.hints.Array,
) -> None:
    jacobian_fwd = jacobian(lambda g: f(Group,g), primal)
    jacobian_numerical = jacnumerical(lambda primal: f(Group,primal))(primal)

    assert_arrays_close(jacobian_fwd, jacobian_numerical, rtol=5e-4, atol=5e-4)


# Exp tests.
def _exp(Group: Type[liegroup.MatrixLieGroup], generator: torch.Tensor) -> torch.Tensor:
    return Group.exp(generator).parameters()


def test_so3_nan():
    """Make sure we don't get NaNs from division when w == 0.

    https://github.com/brentyi/jaxlie/issues/9"""

    # @torch.jit.script
    @torch.autograd.grad
    def func(x):
        return liegroup.SO3.exp(x).log().sum()

    for omega in torch.eye(3) * torch.pi:
        a = omega.float()
        assert all(torch.logical_not(torch.isnan(func(a))))


@general_group_test
def test_exp_random(Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Check that exp Jacobians are consistent, with randomly sampled transforms."""
    del batch_axes  # Not used for autodiff tests.
    generator = torch.randn(Group.tangent_dim).to(device)
    _assert_jacobians_close(Group=Group, f=_exp, primal=generator)


@general_group_test
def test_exp_identity(Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Check that exp Jacobians are consistent, with transforms close to the
    identity."""
    del batch_axes  # Not used for autodiff tests.
    generator = torch.randn(Group.tangent_dim).to(device) * 1e-6
    _assert_jacobians_close(Group=Group, f=_exp, primal=generator)


# Log tests.
def _log(Group: Type[liegroup.MatrixLieGroup], params: torch.Tensor) -> torch.Tensor:
    return Group.log(Group(params))


@general_group_test
def test_log_random(Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Check that log Jacobians are consistent, with randomly sampled transforms."""
    del batch_axes  # Not used for autodiff tests.
    params = Group.exp(torch.randn(Group.tangent_dim).to(device)).parameters()
    _assert_jacobians_close(Group=Group, f=_log, primal=params)


@general_group_test
def test_log_identity(Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Check that log Jacobians are consistent, with transforms close to the
    identity."""
    params = Group.exp(torch.randn(Group.tangent_dim).to(device) * 1e-6).parameters()
    _assert_jacobians_close(Group=Group, f=_log, primal=params)


# Adjoint tests.
def _adjoint(Group: Type[liegroup.MatrixLieGroup], params: torch.Tensor) -> torch.Tensor:
    return Group(params).adjoint().flatten()


@general_group_test
def test_adjoint_random(
    Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    """Check that adjoint Jacobians are consistent, with randomly sampled transforms."""
    del batch_axes  # Not used for autodiff tests.
    params = Group.exp(torch.randn(Group.tangent_dim).to(device)).parameters()
    _assert_jacobians_close(Group=Group, f=_adjoint, primal=params)


@general_group_test
def test_adjoint_identity(
    Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    """Check that adjoint Jacobians are consistent, with transforms close to the
    identity."""
    del batch_axes  # Not used for autodiff tests.
    params = Group.exp(torch.randn(Group.tangent_dim).to(device) * 1e-6).parameters()
    _assert_jacobians_close(Group=Group, f=_adjoint, primal=params)


# Apply tests.
def _apply(Group: Type[liegroup.MatrixLieGroup], params: torch.Tensor) -> torch.Tensor:
    return Group(params) @ torch.ones(Group.space_dim)

@general_group_test
def test_apply_random(Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Check that apply Jacobians are consistent, with randomly sampled transforms."""
    del batch_axes  # Not used for autodiff tests.
    params = Group.exp(torch.randn(Group.tangent_dim).to(device)).parameters()
    _assert_jacobians_close(Group=Group, f=_apply, primal=params)


@general_group_test
def test_apply_identity(
    Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    """Check that apply Jacobians are consistent, with transforms close to the
    identity."""
    del batch_axes  # Not used for autodiff tests.
    params = Group.exp(torch.randn(Group.tangent_dim).to(device) * 1e-6).parameters()
    _assert_jacobians_close(Group=Group, f=_apply, primal=params)


# Multiply tests.
def _multiply(Group: Type[liegroup.MatrixLieGroup], params: torch.Tensor) -> torch.Tensor:
    return Group(params) @ Group(params).parameters()


@general_group_test
def test_multiply_random(
    Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    """Check that multiply Jacobians are consistent, with randomly sampled
    transforms."""
    del batch_axes  # Not used for autodiff tests.
    params = Group.exp(torch.randn(Group.tangent_dim).to(device)).parameters()
    _assert_jacobians_close(Group=Group, f=_multiply, primal=params)


@general_group_test
def test_multiply_identity(
    Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    """Check that multiply Jacobians are consistent, with transforms close to the
    identity."""
    del batch_axes  # Not used for autodiff tests.
    params = Group.exp(torch.randn(Group.tangent_dim).to(device) * 1e-6).parameters()
    _assert_jacobians_close(Group=Group, f=_multiply, primal=params)


# Inverse tests.
def _inverse(Group: Type[liegroup.MatrixLieGroup], params: torch.Tensor) -> torch.Tensor:
    return Group(params).inverse().parameters()


@general_group_test
def test_inverse_random(
    Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    """Check that inverse Jacobians are consistent, with randomly sampled transforms."""
    del batch_axes  # Not used for autodiff tests.
    params = Group.exp(torch.randn(Group.tangent_dim).to(device)).parameters()
    _assert_jacobians_close(Group=Group, f=_inverse, primal=params)


@general_group_test
def test_inverse_identity(
    Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    """Check that inverse Jacobians are consistent, with transforms close to the
    identity."""
    del batch_axes  # Not used for autodiff tests.
    params = Group.exp(torch.randn(Group.tangent_dim).to(device) * 1e-6).parameters()
    _assert_jacobians_close(Group=Group, f=_inverse, primal=params)


if __name__ == "__main__":
    # test_exp_random(liegroup.SO3,batch_axes=())
    # print("SE3 test_exp_random passed")
    # test_exp_random(liegroup.SE2,batch_axes=())
    # print("SE2 test_exp_random passed")
    # test_exp_identity(liegroup.SE3,batch_axes=())
    # print("SE3 test_exp_identity passed")
    test_log_random(liegroup.SE2,batch_axes=())
    print("SE2 test_log_random passed")
    test_log_identity(liegroup.SE3,batch_axes=())
    print("SE3 test_log_identity passed")

