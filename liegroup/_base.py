import abc
from typing import ClassVar, Tuple, TypeVar, Union, Generic

import torch
from torch import nn, Tensor
from typing_extensions import Self, final, get_args

class MatrixLieGroup(abc.ABC):
    """Interface definition for matrix Lie groups."""

    matrix_dim: ClassVar[int]
    parameters_dim: ClassVar[int]
    tangent_dim: ClassVar[int]
    space_dim: ClassVar[int]

    def __init__(self, parameters: Tensor):
        """Construct a group object from its underlying parameters."""
        raise NotImplementedError()

    def __matmul__(self, other: Union[Self, Tensor]) -> Union[Self, Tensor]:
        """Overload for the `@` operator."""
        if isinstance(other, Tensor):
            return self.apply(target=other)
        elif isinstance(other, MatrixLieGroup):
            assert self.space_dim == other.space_dim
            return self.multiply(other=other)
        else:
            raise TypeError(f"Invalid argument type for `@` operator: {type(other)}")

    @classmethod
    @abc.abstractmethod
    def identity(cls, batch_axes: Tuple[int, ...] = ()) -> Self:
        pass

    @classmethod
    @abc.abstractmethod
    def from_matrix(cls, matrix: Tensor) -> Self:
        pass

    @abc.abstractmethod
    def as_matrix(self) -> Tensor:
        pass

    @abc.abstractmethod
    def parameters(self) -> Tensor:
        pass

    @abc.abstractmethod
    def apply(self, target: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def multiply(self, other: Self) -> Self:
        pass

    @classmethod
    @abc.abstractmethod
    def exp(cls, tangent: Tensor) -> Self:
        pass

    @abc.abstractmethod
    def log(self) -> Tensor:
        pass

    @abc.abstractmethod
    def adjoint(self) -> Tensor:
        pass

    @abc.abstractmethod
    def inverse(self) -> Self:
        pass

    @abc.abstractmethod
    def normalize(self) -> Self:
        pass

    @classmethod
    @abc.abstractmethod
    def sample_uniform(cls, key: Tensor, batch_axes: Tuple[int, ...] = ()) -> Self:
        pass

    @final
    def get_batch_axes(self) -> Tuple[int, ...]:
        return self.parameters().shape[:-1]

class SOBase(MatrixLieGroup):
    """Base class for special orthogonal groups."""

ContainedSOType = TypeVar("ContainedSOType", bound=SOBase)

class SEBase(Generic[ContainedSOType], MatrixLieGroup):
    """Base class for special Euclidean groups."""

    @classmethod
    @abc.abstractmethod
    def from_rotation_and_translation(cls, rotation: ContainedSOType, translation: Tensor) -> Self:
        pass

    @final
    @classmethod
    def from_rotation(cls, rotation: ContainedSOType) -> Self:
        return cls.from_rotation_and_translation(
            rotation=rotation,
            translation=torch.zeros((*rotation.get_batch_axes(), cls.space_dim), dtype=rotation.parameters().dtype)
        )

    @final
    @classmethod
    def from_translation(cls, translation: Tensor) -> Self:
        rotation_class = get_args(cls.__orig_bases__[0])[0]
        return cls.from_rotation_and_translation(
            rotation=rotation_class.identity(),
            translation=translation
        )

    @abc.abstractmethod
    def rotation(self) -> ContainedSOType:
        pass

    @abc.abstractmethod
    def translation(self) -> Tensor:
        pass

    @final
    def apply(self, target: Tensor) -> Tensor:
        return self.rotation() @ target + self.translation()

    @final
    def multiply(self, other: Self) -> Self:
        return type(self).from_rotation_and_translation(
            rotation=self.rotation() @ other.rotation(),
            translation=(self.rotation() @ other.translation()) + self.translation()
        )

    @final
    def inverse(self) -> Self:
        R_inv = self.rotation().inverse()
        return type(self).from_rotation_and_translation(
            rotation=R_inv,
            translation=-(R_inv @ self.translation())
        )

    @final
    def normalize(self) -> Self:
        return type(self).from_rotation_and_translation(
            rotation=self.rotation().normalize(),
            translation=self.translation()
        )