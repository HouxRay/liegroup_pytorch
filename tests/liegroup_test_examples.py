"""Tests with explicit examples."""

import os
import sys
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0,pythonpath)
import liegroup
import torch
from torch import nn
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.liegroup_test_utils import assert_arrays_close, assert_transforms_close, sample_transform


device = "cuda" if torch.cuda.is_available() else "cpu"

@settings(deadline=None)
@given(_random_module=st.random_module())
def test_se2_translation(_random_module):
    """Simple test for SE(2) translation terms."""
    translation = torch.randn(2).to(device)
    T = liegroup.SE2.from_xy_theta(*translation, theta=torch.tensor(0.0))
    assert_arrays_close(T @ translation, translation * 2)
    

@settings(deadline=None)
@given(_random_module=st.random_module())
def test_se3_translation(_random_module):
    """Simple test for SE(3) translation terms."""
    translation = torch.randn(3).to(device)
    T = liegroup.SE3.from_rotation_and_translation(
        rotation=liegroup.SO3.identity(),
        translation=translation,
    )
    assert_arrays_close(T @ translation, translation * 2)

def test_se2_rotation():
    """Simple test for SE(2) rotation terms."""
    T_w_b = liegroup.SE2.from_rotation_and_translation(
        rotation=liegroup.SO2.from_radians(torch.tensor(torch.pi / 2.0)),
        translation=torch.zeros(2),
    )
    p_b = torch.tensor([1.0, 0.0]).to(device)
    p_w = torch.tensor([0.0, 1.0]).to(device)
    assert_arrays_close(T_w_b @ p_b, p_w) 


def test_se3_rotation():
    """Simple test for SE(3) rotation terms."""
    T_w_b = liegroup.SE3.from_rotation_and_translation(
        rotation=liegroup.SO3.from_rpy_radians(torch.tensor(torch.pi / 2.0).to(device), 
                                                 torch.tensor(0.0).to(device), 
                                                 torch.tensor(0.0).to(device)),
        translation=torch.zeros(3),
    )
    T_w_b_alt = liegroup.SE3.from_rotation(
        liegroup.SO3.from_rpy_radians(torch.tensor(torch.pi / 2.0).to(device), 
                                                 torch.tensor(0.0).to(device), 
                                                 torch.tensor(0.0).to(device)),
    )
    p_b = torch.tensor([0.0, 1.0, 0.0]).to(device)
    p_w = torch.tensor([0.0, 0.0, 1.0]).to(device)
    assert_arrays_close(T_w_b @ p_b, T_w_b_alt @ p_b, p_w)   

def test_se3_from_translation():
    """Simple test for SE(3) rotation terms."""
    T_w_b = liegroup.SE3.from_rotation_and_translation(
        rotation=liegroup.SO3.identity(),
        translation=torch.arange(3) * 1.0,
    )
    T_w_b_alt = liegroup.SE3.from_translation(torch.arange(3) * 1.0)
    p_b = torch.tensor([0.0, 1.0, 0.0]).to(device)
    p_w = torch.tensor([0.0, 2.0, 2.0]).to(device)
    assert_arrays_close(T_w_b @ p_b, T_w_b_alt @ p_b, p_w)  

def test_so3_xyzw_basic():
    """Check that we can create an SO3 object from an xyzw quaternion."""
    assert_transforms_close(
        liegroup.SO3.from_quaternion_xyzw(torch.tensor([0.0, 0.0, 0.0, 1.0]).to(device)),
        liegroup.SO3.identity(),
    )   

@settings(deadline=None)
@given(_random_module=st.random_module())
def test_se3_compose(_random_module):
    """Compare SE3 composition in matrix form vs compact form."""
    T1 = sample_transform(liegroup.SE3)
    T2 = sample_transform(liegroup.SE3)
    assert_arrays_close(T1.as_matrix() @ T2.as_matrix(), (T1 @ T2).as_matrix())
    assert_transforms_close(
        liegroup.SE3.from_matrix(T1.as_matrix() @ T2.as_matrix()), T1 @ T2
    )


if __name__ == "__main__":
    test_se3_translation()
    print("se3_translation passed")
    test_se2_translation()
    print("se2_translation passed")
    test_se3_rotation()
    print("se2_rotation passed")
    test_se2_rotation()
    print("se3_rotation passed")
    test_se3_from_translation()
    print("se3_from_translation passed")
    test_so3_xyzw_basic()
    print("so3_xyzw_basic passed")
    test_se3_compose()
    print("test_se3_compose passed")