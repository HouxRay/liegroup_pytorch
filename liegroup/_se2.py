import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple
from . import _base, hints
from ._so2 import SO2
from .utils import broadcast_leading_axes, get_epsilon,register_lie_group  # You'll need to adjust these utilities for PyTorch
#import Tuple

@register_lie_group(
    matrix_dim=3,
    parameters_dim=4,
    tangent_dim=3,
    space_dim=2,
)

@dataclass
class SE2(_base.SEBase[SO2]):
    """Special Euclidean group for proper rigid transforms in 2D. Broadcasting
    rules are the same as for numpy.

    Internal parameterization is `(cos, sin, x, y)`. Tangent parameterization is `(vx, vy, omega)`.
    """
    unit_complex_xy: torch.Tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """Internal parameters. `(cos, sin, x, y)`. Shape should be `(*, 4)`."""

    def __repr__(self) -> str:
        unit_complex = torch.round(self.unit_complex_xy[..., :2], decimals=5).to(self.device)
        xy = torch.round(self.unit_complex_xy[..., 2:], decimals=5).to(self.device)
        return f"{self.__class__.__name__}(unit_complex={unit_complex}, xy={xy})"

    @staticmethod
    def from_xy_theta(x: hints.Scalar, y: hints.Scalar, theta: hints.Scalar) -> "SE2":
        cos = torch.cos(theta).to(SE2.device)
        sin = torch.sin(theta).to(SE2.device)
        return SE2(unit_complex_xy=torch.stack([cos, sin, x, y], dim=-1))

    @classmethod
    def from_rotation_and_translation(cls, rotation: SO2, translation: hints.Array) -> "SE2":
        assert translation.shape[-1:] == (2,)
        rotation, translation = broadcast_leading_axes((rotation, translation))  # Adjust for PyTorch
        return SE2(
            unit_complex_xy=torch.cat([rotation.unit_complex, translation], dim=-1)
        )

    def rotation(self) -> SO2:
        return SO2(unit_complex=self.unit_complex_xy[..., :2].to(self.device))

    def translation(self) -> torch.Tensor:
        return self.unit_complex_xy[..., 2:].to(self.device)

    @classmethod
    def identity(cls, batch_axes: Tuple[int, ...] = ()) -> "SE2":
        return SE2(
            unit_complex_xy=torch.broadcast_to(
                torch.tensor([1.0, 0.0, 0.0, 0.0]), (*batch_axes, 4)
            ).to(cls.device)
        )

    @classmethod
    def from_matrix(cls, matrix: hints.Array) -> "SE2":
        assert matrix.shape[-2:] == (3, 3)
        return SE2.from_rotation_and_translation(
            rotation=SO2.from_matrix(matrix[..., :2, :2]),
            translation=matrix[..., :2, 2],
        )
    
    
    def parameters(self) -> torch.Tensor:
        return self.unit_complex_xy

    def as_matrix(self) -> torch.Tensor:
        cos, sin, x, y = torch.moveaxis(self.unit_complex_xy, -1, 0).to(self.device)
        out = torch.stack([
            cos,
            -sin,
            x,
            sin,
            cos,
            y,
            torch.zeros_like(x),
            torch.zeros_like(x),
            torch.ones_like(x),
        ], dim=-1).reshape((*self.unit_complex_xy.shape[:-1], 3, 3)).to(self.device)
        return out

    @classmethod
    def exp(cls, tangent: hints.Array) -> "SE2":
        # Assume get_epsilon and SO2.from_radians are defined elsewhere
        assert tangent.shape[-1:] == (3,)
        tangent = tangent.to(SO2.device)
        theta = tangent[..., 2]
        use_taylor = torch.abs(theta) < get_epsilon(tangent.dtype)
        
        # Handling where conditions
        safe_theta = torch.where(
            use_taylor,
            torch.ones_like(theta),
            theta
        ).to(cls.device)

        theta_sq = theta**2
        sin_over_theta = torch.where(
            use_taylor,
            1.0 - theta_sq / 6.0,
            torch.sin(safe_theta) / safe_theta
        ).to(cls.device)
        one_minus_cos_over_theta = torch.where(
            use_taylor,
            0.5 * theta - theta * theta_sq / 24.0,
            (1.0 - torch.cos(safe_theta)) / safe_theta
        ).to(cls.device)

        V = torch.stack(
            [
                sin_over_theta,
                -one_minus_cos_over_theta,
                one_minus_cos_over_theta,
                sin_over_theta
            ],
            dim=-1
        ).reshape(*tangent.shape[:-1], 2, 2).to(cls.device)
        return cls.from_rotation_and_translation(
            rotation=SO2.from_radians(theta),
            translation=torch.einsum("...ij,...j->...i", V, tangent[..., :2])
        )

    def log(self) -> torch.Tensor:
        theta = self.rotation().log()[..., 0]
        
        cos = torch.cos(theta).to(self.device)
        cos_minus_one = cos - 1.0
        half_theta = theta / 2.0
        use_taylor = torch.abs(cos_minus_one) < get_epsilon(theta.dtype)
        
        safe_cos_minus_one = torch.where(
            use_taylor,
            torch.ones_like(cos_minus_one),
            cos_minus_one
        ).to(self.device)
        
        half_theta_over_tan_half_theta = torch.where(
            use_taylor,
            1.0 - theta**2 / 12.0,
            -(half_theta * torch.sin(theta)) / safe_cos_minus_one
        ).to(self.device)

        V_inv = torch.stack(
            [
                half_theta_over_tan_half_theta,
                half_theta,
                -half_theta,
                half_theta_over_tan_half_theta
            ],
            dim=-1
        ).reshape(*theta.shape, 2, 2).to(self.device)

        tangent = torch.cat(
            [
                torch.einsum("...ij,...j->...i", V_inv, self.translation()),
                theta[..., None]
            ],
            dim=-1
        ).to(self.device)
        return tangent

    def adjoint(self:"SE2") -> torch.Tensor:
        cos, sin, x, y = torch.moveaxis(self.unit_complex_xy, -1, 0)
        return torch.stack(
            [
                cos,
                -sin,
                y,
                sin,
                cos,
                -x,
                torch.zeros_like(x),
                torch.zeros_like(x),
                torch.ones_like(x)
            ],
            dim=-1
        ).reshape(*self.get_batch_axes(), 3, 3).to(self.device)

    @classmethod
    def sample_uniform(cls, key:int, batch_axes=()) -> "SE2":
        torch.manual_seed(key)
        return cls.from_rotation_and_translation(
            rotation=SO2.sample_uniform(key, batch_axes=batch_axes),
            translation=torch.rand(
                (*batch_axes, 2),
                dtype=torch.float32
            ).to(cls.device) * 2 - 1.0
        )

if __name__ == "__main__":
    # Example usage:
    se2 = SO2.from_radians(torch.tensor(1.2))
    print(se2)
    print(se2.as_matrix())
