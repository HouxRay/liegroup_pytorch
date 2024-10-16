"""Tests for general operation definitions."""

from typing import Tuple, Type


import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.liegroup_test_utils import (
    assert_arrays_close,
    assert_transforms_close,
    general_group_test,
    sample_transform,
)

import liegroup

device='cuda' if torch.cuda.is_available() else 'cpu'

@general_group_test
def test_sample_uniform_valid(
    Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    """Check that sample_uniform() returns valid group members."""
    T = sample_transform(Group, batch_axes)  # Calls sample_uniform under the hood.
    assert_transforms_close(T, T.normalize())


@settings(deadline=None)
@given(_random_module=st.random_module())
def test_so2_from_to_radians_bijective(_random_module):
    """Check that we can convert from and to radians."""
    radians = torch.rand(1).to(device) * 2 * torch.pi - torch.pi
    assert_arrays_close(liegroup.SO2.from_radians(radians).as_radians(), radians)


@settings(deadline=None)
@given(_random_module=st.random_module())
def test_so3_xyzw_bijective(_random_module):
    """Check that we can convert between xyzw and wxyz quaternions."""
    T = sample_transform(liegroup.SO3)
    assert_transforms_close(T, liegroup.SO3.from_quaternion_xyzw(T.as_quaternion_xyzw()))


@settings(deadline=None)
@given(_random_module=st.random_module())
def test_so3_rpy_bijective(_random_module):
    """Check that we can convert between quaternions and Euler angles."""
    T = sample_transform(liegroup.SO3)
    assert_transforms_close(T, liegroup.SO3.from_rpy_radians(*T.as_rpy_radians()))


@general_group_test
def test_log_exp_bijective(
    Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    """Check 1-to-1 mapping for log <=> exp operations."""
    transform = sample_transform(Group, batch_axes)

    tangent = transform.log()
    assert tangent.shape == (*batch_axes, Group.tangent_dim)

    exp_transform = Group.exp(tangent)
    assert_transforms_close(transform, exp_transform)
    assert_arrays_close(tangent, exp_transform.log())


@general_group_test
def test_inverse_bijective(
    Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    """Check inverse of inverse."""
    transform = sample_transform(Group, batch_axes)
    assert_transforms_close(transform, transform.inverse().inverse())


@general_group_test
def test_matrix_bijective(
    Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    """Check that we can convert to and from matrices."""
    transform = sample_transform(Group, batch_axes)
    assert_transforms_close(transform, Group.from_matrix(transform.as_matrix()))


@general_group_test
def test_adjoint(Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Check adjoint definition."""
    transform = sample_transform(Group, batch_axes)
    omega = torch.randn(*batch_axes, Group.tangent_dim).to(device)
    assert_transforms_close(
        transform @ Group.exp(omega),
        Group.exp(torch.einsum("...ij,...j->...i", transform.adjoint(), omega))
        @ transform,
    )


@general_group_test
def test_repr(Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Smoke test for __repr__ implementations."""
    transform = sample_transform(Group, batch_axes)
    print(transform)


@general_group_test
def test_apply(Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Check group action interfaces."""
    T_w_b = sample_transform(Group, batch_axes)
    p_b = torch.randn(*batch_axes, Group.space_dim).to(device)

    if Group.matrix_dim == Group.space_dim:
        assert_arrays_close(
            T_w_b @ p_b,
            T_w_b.apply(p_b),
            torch.einsum("...ij,...j->...i", T_w_b.as_matrix(), p_b),
        )
    else:
        # Homogeneous coordinates.
        assert Group.matrix_dim == Group.space_dim + 1
        assert_arrays_close(
            T_w_b @ p_b,
            T_w_b.apply(p_b),
            torch.einsum(
                "...ij,...j->...i",
                T_w_b.as_matrix(),
                torch.cat([p_b, torch.ones_like(p_b[..., :1])], dim=-1),
            )[..., :-1],
        )


@general_group_test
def test_multiply(Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Check multiply interfaces."""
    T_w_b = sample_transform(Group, batch_axes)
    T_b_a = sample_transform(Group, batch_axes)
    assert_arrays_close(
        torch.einsum(
            "...ij,...jk->...ik", T_w_b.as_matrix(), T_w_b.inverse().as_matrix()
        ),
        torch.broadcast_to(
            torch.eye(Group.matrix_dim).to(device), (*batch_axes, Group.matrix_dim, Group.matrix_dim)
        ),
    )
    assert_arrays_close(
        torch.einsum(
            "...ij,...jk->...ik", T_w_b.as_matrix(), torch.linalg.inv(T_w_b.as_matrix())
        ),
        torch.broadcast_to(
            torch.eye(Group.matrix_dim).to(device), (*batch_axes, Group.matrix_dim, Group.matrix_dim)
        ),
    )
    assert_transforms_close(T_w_b @ T_b_a, Group.multiply(T_w_b, T_b_a))

if __name__ == "__main__":
    test_sample_uniform_valid(liegroup.SE2,batch_axes=())
    print("SE2 test_sample_uniform_valid passed")
    test_sample_uniform_valid(liegroup.SE3,batch_axes=())
    print("SE3 test_sample_uniform_valid passed")
    test_so2_from_to_radians_bijective()
    print("test_so2_from_to_radians_bijective passed")
    test_so3_xyzw_bijective()
    print("test_so3_xyzw_bijective passed")
    test_so3_rpy_bijective()
    print("test_so3_rpy_bijective passed")
    test_log_exp_bijective(liegroup.SE2,batch_axes=())
    print("SE2 test_log_exp_bijective passed")
    test_log_exp_bijective(liegroup.SE3,batch_axes=())
    print("SE3 test_log_exp_bijective passed")
    test_inverse_bijective(liegroup.SE2,batch_axes=())
    print("SE2 test_inverse_bijective passed")
    test_inverse_bijective(liegroup.SE3,batch_axes=())
    print("SE3 test_inverse_bijective passed")
    test_matrix_bijective(liegroup.SE2,batch_axes=())
    print("SE2 test_matrix_bijective passed")
    test_matrix_bijective(liegroup.SE3,batch_axes=())
    print("SE3 test_matrix_bijective passed")
    test_adjoint(liegroup.SE2,batch_axes=())
    print("SE2 test_adjoint passed")
    test_adjoint(liegroup.SE3,batch_axes=())
    print("SE3 test_adjoint passed")
    test_repr(liegroup.SE2,batch_axes=())
    print("SE2 test_repr passed")
    test_repr(liegroup.SE3,batch_axes=())
    print("SE3 test_repr passed")
    test_apply(liegroup.SE2,batch_axes=())
    test_apply(liegroup.SE3,batch_axes=())
    test_multiply(liegroup.SE2,batch_axes=())
    test_multiply(liegroup.SE3,batch_axes=())

