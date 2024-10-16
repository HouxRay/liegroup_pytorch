"""Test transform serialization, for things like saving calibrated transforms to
disk."""

from typing import Tuple, Type
import torch
import io

from tests.liegroup_test_utils import assert_transforms_close, general_group_test, sample_transform

import liegroup


@general_group_test
def test_serialization_state_dict_bijective(
    Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    """Check bijectivity of state dict representation conversions."""
    T = sample_transform(Group, batch_axes)
    T_path = "T.pt"
    torch.save(T,T_path)
    T_recovered = torch.load(T_path)
    assert_transforms_close(T, T_recovered)


@general_group_test
def test_serialization_bytes_bijective(
    Group: Type[liegroup.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    """Check bijectivity of byte representation conversions."""
    T = sample_transform(Group, batch_axes)
    buffer = io.BytesIO()
    torch.save(T, buffer)
    T_recovered = torch.load(io.BytesIO(buffer.getvalue()))
    assert_transforms_close(T, T_recovered)


if __name__ == "__main__":
    test_serialization_state_dict_bijective(liegroup.SE3,batch_axes=())
    print("SE3 test_serialization_state_dict_bijective passed")
    test_serialization_state_dict_bijective(liegroup.SE2,batch_axes=())
    print("SE2 test_serialization_state_dict_bijective passed")  
    test_serialization_bytes_bijective(liegroup.SE3,batch_axes=())  
    print("SE3 test_serialization_bytes_bijective passed")  
    test_serialization_bytes_bijective(liegroup.SE2,batch_axes=())  
    print("SE2 test_serialization_bytes_bijective passed")  