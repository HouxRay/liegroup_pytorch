import functools
import random
import torch
from torch import nn
from hypothesis import given, settings
from hypothesis import strategies as st
import pytest
import scipy.optimize


from typing import Any, Callable, List, Tuple, Type, TypeVar
import liegroup


# Run all tests with double-precision.

T = TypeVar("T", bound=liegroup.MatrixLieGroup)
device = "cuda" if torch.cuda.is_available() else "cpu"

def sample_transform(Group: Type[T], batch_axes: Tuple[int, ...] = ()) -> T:
    """Sample a random transform from a group."""
    seed = random.getrandbits(32)
    strategy = random.randint(0,2)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if strategy == 0:
        # Uniform sampling.
        return Group.sample_uniform(key = round(torch.randn(1).item()),
                                    batch_axes=batch_axes)
    
    elif strategy == 1:
        #  Sample from normally-sampled tangent vector.
        return Group.exp(torch.randn(*batch_axes, 
                                     Group.tangent_dim).to(device))
    
    elif strategy == 2:
        # Sample near identity.
        return Group.exp(torch.randn(*batch_axes, 
                                     Group.tangent_dim).to(device) * 1e-7)
    
    else:
        return False
    

def general_group_test(
    f: Callable[[Type[liegroup.MatrixLieGroup], Tuple[int, ...]], None],
    max_examples: int = 30,
) -> Callable[[Type[liegroup.MatrixLieGroup], Tuple[int, ...], Any], None]:
    """Decorator for defining tests that run on all group types."""

    # Disregard unused argument.
    def f_wrapped(Group:Type[liegroup.MatrixLieGroup],
                  batch_axes: Tuple[int, ...], _random_module
    ) -> None:
        f(Group, batch_axes)

    # Disable timing check (first run requires JIT tracing and will be slower).
    f_wrapped = settings(deadline=None, max_examples=max_examples)(f_wrapped)

    # Add _random_module parameter.
    f_wrapped = given(_random_module=st.random_module())(f_wrapped)

    # Parametrize tests with each group type.
    f_wrapped = pytest.mark.parametrize(
        "Group",
        [
            liegroup.SO2,
            liegroup.SE2,
            liegroup.SO3,
            liegroup.SE3,
        ],
    )(f_wrapped)

    # Parametrize tests with each group type.
    f_wrapped = pytest.mark.parametrize(
        "batch_axes",
        [
            (),
            (1,),
            (3, 1, 2, 1),
        ],
    )(f_wrapped)
    return f_wrapped

general_group_test_faster = functools.partial(general_group_test, max_examples=5)

def assert_transforms_close(a: liegroup.MatrixLieGroup, b: liegroup.MatrixLieGroup):
    """Make sure two transforms are equivalent."""
    # Check matrix representation.
    assert_arrays_close(a.as_matrix(), b.as_matrix())

    # Flip signs for quaternions.
    # We use `torch.asarray` here in case inputs are onp arrays and don't support `.at()`.
    p1 = torch.asarray(a.parameters())
    p2 = torch.asarray(b.parameters())
    if isinstance(a, liegroup.SO3):
        p1 = p1 * torch.sign(torch.sum(p1, dim=-1, keepdims=True)).to(device)
        p2 = p2 * torch.sign(torch.sum(p2, dim=-1, keepdims=True)).to(device)
    elif isinstance(a, liegroup.SE3):
        p1 = p1[..., :4] * torch.sign(torch.sum(p1[..., :4], dim=-1, keepdim=True)).to(device)
        p2 = p2[..., :4] * torch.sign(torch.sum(p2[..., :4], dim=-1, keepdim=True)).to(device)

    # Make sure parameters are equal.
    assert_arrays_close(p1, p2)

def assert_arrays_close(
    *arrays: liegroup.hints.Array,
    rtol: float = 1e-4,
    atol: float = 1e-4,
):
    """Make sure two arrays are close. (and not NaN)"""
    for array1, array2 in zip(arrays[:-1], arrays[1:]):
        torch.testing.assert_allclose(array1, array2, rtol=rtol, atol=atol)
        assert not torch.any(torch.isnan(array1))
        assert not torch.any(torch.isnan(array2))

def jacnumerical(
    f: Callable[[liegroup.hints.Array], torch.Tensor],
) -> Callable[[liegroup.hints.Array], torch.Tensor]:
    """Decorator for computing numerical Jacobians of vector->vector functions."""

    def wrapped(primal: liegroup.hints.Array) -> torch.Tensor:
        output_dim: int
        (output_dim,) = f(primal).shape
        primal_j = primal.cpu().detach().numpy()
        jacobian_rows: List[torch.Tensor] = []
        for i in range(output_dim):
            jacobian_row: torch.Tensor = scipy.optimize.approx_fprime(
                primal_j, lambda p: f(torch.from_numpy(p).to(device))[i].item(), epsilon=1e-5
            )
            assert jacobian_row.shape == primal.shape
            jacobian_rows.append(torch.from_numpy(jacobian_row).to(device))

        return torch.stack(jacobian_rows, dim=0)

    return wrapped