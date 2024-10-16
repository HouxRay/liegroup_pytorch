from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple, Union, overload

import torch
from typing_extensions import ParamSpec

from .._base import MatrixLieGroup
from . import _deltas, _tree_utils
from torch.autograd.functional import jacobian



def zero_tangents(pytree: Any) -> _tree_utils.TangentPytree:
    """Replace all values in a Pytree with zero vectors on the corresponding tangent
    spaces."""

    def tangent_zero(t: MatrixLieGroup) -> torch.Tensor:
        return torch.zeros(t.get_batch_axes() + (t.tangent_dim,))

    return _tree_utils._map_group_trees(
        tangent_zero,
        lambda array: torch.zeros_like(array),
        pytree,
    )


AxisName = Any

P = ParamSpec("P")

@overload
def grad(
    fun: Callable[P, Any],
    argnums: int = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[P, _tree_utils.TangentPytree]: ...

@overload
def grad(
    fun: Callable[P, Any],
    argnums: Sequence[int],
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[P, Tuple[_tree_utils.TangentPytree, ...]]: ...

def grad(
    fun: Callable[P, Any],
    argnums: Union[int, Sequence[int]] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
):
    """Same as `jax.grad`, but computes gradients of Lie groups with respect to
    tangent spaces."""

    compute_value_and_grad = value_and_grad(
        fun=fun,
        argnums=argnums,
        has_aux=has_aux,
    )

    def grad_fun(*args, **kwargs):
        ret = compute_value_and_grad(*args, **kwargs)
        if has_aux:
            return ret[1], ret[0][1]
        else:
            return ret[1]

    return grad_fun

@overload
def value_and_grad(
    fun: Callable[P, Any],
    argnums: int = 0,
    has_aux: bool = False,
) -> Callable[P, Tuple[Any, _tree_utils.TangentPytree]]: ...


@overload
def value_and_grad(
    fun: Callable[P, Any],
    argnums: Sequence[int],
    has_aux: bool = False,
) -> Callable[P, Tuple[Any, Tuple[_tree_utils.TangentPytree, ...]]]: ...

def value_and_grad(
    fun: Callable[P, Any],
    argnums: Union[int, Sequence[int]] = 0.0,
    has_aux: bool = False,
):
    """Same as `jax.value_and_grad`, but computes gradients of Lie groups with respect to
    tangent spaces."""

    def wrapped_grad(*args, **kwargs):
        def tangent_fun(*tangent_args, **tangent_kwargs):
            return fun(  # type: ignore
                *_deltas.rplus(args, tangent_args),
                **_deltas.rplus(kwargs, tangent_kwargs),
            )
        # Put arguments onto tangent space.
        tangent_args = [zero_tangents(arg) for arg in args]
        tangent_args[0].requires_grad_()
        tangent_kwargs = {k: zero_tangents(v) for k, v in kwargs.items()}
        output = tangent_fun(*tangent_args, **tangent_kwargs)

        if has_aux:
            value, aux = output if isinstance(output, tuple) else (output, None)
        else:
            value, aux = output, None        
        value.requires_grad
        # 计算梯度
        if isinstance(value, torch.Tensor):
            grads = torch.autograd.grad(value, tangent_args, retain_graph=True)

            if isinstance(argnums, int):
                return value, grads[argnums] if grads else None
            elif isinstance(argnums, (list, tuple)):
                return value, [grads[i] for i in argnums if grads and i < len(grads)]

        return value, None  # 返回值和 None 作为梯度
        # return torch.autograd.grad(
        #     tangent_fun(*tangent_args, **tangent_kwargs),
        #     torch.tensor([argnums],dtype=float,requires_grad=True),
        # )(*tangent_args, **tangent_kwargs)
    
        return jacobian(
            tangent_fun,
            *tangent_args,
        )

    return wrapped_grad  # type: ignore
