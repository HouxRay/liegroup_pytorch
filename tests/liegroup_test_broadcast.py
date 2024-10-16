"""Shape tests for broadcasting."""

from typing import Tuple, Type

from hypothesis import given, settings
from hypothesis import strategies as st
import torch
from tests.liegroup_test_utils import (
    assert_arrays_close,
    assert_transforms_close,
    general_group_test,
    sample_transform,
)

import liegroup


@general_group_test
def test_broadcast_multiply(
    Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    if batch_axes == ():
        return

    T = sample_transform(Group, batch_axes) @ sample_transform(Group)
    assert T.get_batch_axes() == batch_axes

    T = sample_transform(Group, batch_axes) @ sample_transform(Group, batch_axes=(1,))
    assert T.get_batch_axes() == batch_axes

    T = sample_transform(Group, batch_axes) @ sample_transform(
        Group, batch_axes=(1,) * len(batch_axes)
    )
    assert T.get_batch_axes() == batch_axes

    T = sample_transform(Group) @ sample_transform(Group, batch_axes)
    assert T.get_batch_axes() == batch_axes

    T = sample_transform(Group, batch_axes=(1,)) @ sample_transform(Group, batch_axes)
    assert T.get_batch_axes() == batch_axes


@general_group_test
def test_broadcast_apply(
    Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    if batch_axes == ():
        return

    T = sample_transform(Group, batch_axes)
    points = torch.randn(Group.space_dim)
    assert (T @ points).shape == (*batch_axes, Group.space_dim)

    T = sample_transform(Group, batch_axes)
    points = torch.randn(1, Group.space_dim)
    assert (T @ points).shape == (*batch_axes, Group.space_dim)

    T = sample_transform(Group, batch_axes)
    points = torch.randn(*((1,) * len(batch_axes)), Group.space_dim)
    assert (T @ points).shape == (*batch_axes, Group.space_dim)

    T = sample_transform(Group)
    points = torch.randn(*batch_axes, Group.space_dim)
    assert (T @ points).shape == (*batch_axes, Group.space_dim)

    T = sample_transform(Group, batch_axes=(1,))
    points = torch.randn(*batch_axes, Group.space_dim)
    assert (T @ points).shape == (*batch_axes, Group.space_dim)


if __name__ == "__main__":
    test_broadcast_multiply(liegroup.SE2,batch_axes=(1,))