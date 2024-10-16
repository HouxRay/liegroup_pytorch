"""Tests for group axioms.

https://proofwiki.org/wiki/Definition:Group_Axioms
"""

from typing import Tuple, Type

import torch
from tests.liegroup_test_utils import (
    assert_arrays_close,
    assert_transforms_close,
    general_group_test,
    sample_transform,
)

import liegroup

device = "cuda" if torch.cuda.is_available() else 'cpu'

@general_group_test
def test_closure(Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Check closure property."""
    transform_a = sample_transform(Group, batch_axes)
    transform_b = sample_transform(Group, batch_axes)

    composed = transform_a @ transform_b
    assert_transforms_close(composed, composed.normalize())
    composed = transform_b @ transform_a
    assert_transforms_close(composed, composed.normalize())
    composed = Group.multiply(transform_a, transform_b)
    assert_transforms_close(composed, composed.normalize())
    composed = Group.multiply(transform_b, transform_a)
    assert_transforms_close(composed, composed.normalize())


@general_group_test
def test_identity(Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Check identity property."""
    transform = sample_transform(Group, batch_axes)
    identity = Group.identity(batch_axes)
    assert_transforms_close(transform, identity @ transform)
    assert_transforms_close(transform, transform @ identity)
    assert_arrays_close(
        transform.as_matrix(),
        torch.einsum("...ij,...jk->...ik", identity.as_matrix(), transform.as_matrix()),
    )
    assert_arrays_close(
        transform.as_matrix(),
        torch.einsum("...ij,...jk->...ik", transform.as_matrix(), identity.as_matrix()),
    )


@general_group_test
def test_inverse(Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Check inverse property."""
    transform = sample_transform(Group, batch_axes)
    identity = Group.identity(batch_axes)
    assert_transforms_close(identity, transform @ transform.inverse())
    assert_transforms_close(identity, transform.inverse() @ transform)
    assert_transforms_close(identity, Group.multiply(transform, transform.inverse()))
    assert_transforms_close(identity, Group.multiply(transform.inverse(), transform))
    assert_arrays_close(
        torch.broadcast_to(
            torch.eye(Group.matrix_dim).to(device), (*batch_axes, Group.matrix_dim, Group.matrix_dim)
        ),
        torch.einsum(
            "...ij,...jk->...ik",
            transform.as_matrix(),
            transform.inverse().as_matrix(),
        ),
    )
    assert_arrays_close(
        torch.broadcast_to(
            torch.eye(Group.matrix_dim).to(device), (*batch_axes, Group.matrix_dim, Group.matrix_dim)
        ),
        torch.einsum(
            "...ij,...jk->...ik",
            transform.inverse().as_matrix(),
            transform.as_matrix(),
        ),
    )


@general_group_test
def test_associative(Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Check associative property."""
    transform_a = sample_transform(Group, batch_axes)
    transform_b = sample_transform(Group, batch_axes)
    transform_c = sample_transform(Group, batch_axes)
    assert_transforms_close(
        (transform_a @ transform_b) @ transform_c,
        transform_a @ (transform_b @ transform_c),
    )


if __name__ == "__main__":
    test_closure(liegroup.SE3, batch_axes=())
    print("SE3 test_closure passed")
    test_closure(liegroup.SE2, batch_axes=())
    print("SE2 test_closure passed")
    test_identity(liegroup.SE2, batch_axes=())
    print("SE2 test_identity passed")
    test_identity(liegroup.SE3, batch_axes=())
    print("SE3 test_identity passed")
    test_inverse(liegroup.SE2, batch_axes=())
    print("SE2 test_inverse passed")
    test_inverse(liegroup.SE3, batch_axes=())
    print("SE3 test_inverse passed")
    test_associative(liegroup.SE2, batch_axes=())
    print("SE2 test_associative passed")
    test_associative(liegroup.SE3, batch_axes=())
    print("SE3 test_associative passed")    
