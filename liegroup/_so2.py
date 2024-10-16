import torch
from torch import nn
from . import _base, hints
from typing import Tuple
from .utils import broadcast_leading_axes, register_lie_group

@register_lie_group(
    matrix_dim=2,
    parameters_dim=2,
    tangent_dim=1,
    space_dim=2,
)

class SO2(_base.SOBase):
    """Special orthogonal group for 2D rotations in PyTorch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __init__(self, unit_complex: torch.Tensor):
        # super(SO2, self).__init__()
        self.unit_complex = unit_complex.to(SO2.device)  # Shape should be (*, 2) where * represents batch dimensions
    
    def __repr__(self):
        unit_complex = torch.round(self.unit_complex, decimals=5).to(self.device)
        return f"{self.__class__.__name__}(unit_complex={unit_complex})"

    @staticmethod
    def from_radians(theta):
        """Construct a rotation object from a scalar angle."""
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        return SO2(torch.stack([cos, sin], dim=-1).to(SO2.device))

    def as_radians(self):
        """Compute a scalar angle from a rotation object."""
        radians = torch.atan2(self.unit_complex[..., 1], self.unit_complex[..., 0]).to(SO2.device)
        return radians

    @classmethod
    def identity(cls, batch_shape=()):
        """Create an identity rotation."""
        return cls(torch.stack([
            torch.ones(batch_shape),
            torch.zeros(batch_shape)
        ], dim=-1)).to(cls.device)

    @classmethod
    def from_matrix(cls, matrix):
        """Construct from a 2x2 rotation matrix."""
        assert matrix.shape[-2:] == (2, 2)
        return cls(matrix[..., :, 0])

    def as_matrix(self):
        """Return the rotation matrix."""
        cos, sin = self.unit_complex.unbind(-1)
        return torch.stack([
            torch.stack([cos, -sin], dim=-1),
            torch.stack([sin, cos], dim=-1)
        ], dim=-2).to(self.device)
    
    def parameters(self) -> torch.Tensor:
        return self.unit_complex

    def apply(self, target):
        """Apply this rotation to a 2D point or batch of points."""
        # self, target = broadcast_leading_axes((self, target))
        return torch.einsum('...ij,...j->...i', self.as_matrix(), target).to(self.device)

    def multiply(self, other):
        """Multiply two rotations."""
        result = torch.einsum('...ij,...j->...i', self.as_matrix(), other.unit_complex).to(self.device)
        return SO2(result)

    def exp(tangent):
        """Exponential map from tangent space to the group."""
        theta = tangent[..., 0]
        return SO2.from_radians(theta)

    def log(self):
        """Logarithmic map from the group to the tangent space."""
        return torch.atan2(self.unit_complex[..., 1], self.unit_complex[..., 0]).unsqueeze(-1).to(self.device)

    def adjoint(self):
        """Adjoint matrix (identity for SO2)."""
        return torch.ones(self.unit_complex.shape[:-1] + (1, 1)).to(self.device)

    def inverse(self):
        """Inverse of the rotation (transpose for SO2)."""
        return SO2(self.unit_complex * torch.tensor([1, -1]).to(self.device))

    def normalize(self):
        """Normalize the unit complex vector."""
        norm = torch.linalg.norm(self.unit_complex, dim=-1, keepdim=True)
        return SO2(self.unit_complex / norm)

    @classmethod
    def sample_uniform(
        cls, key: int, batch_axes = ()
        ):
        """Sample a uniform random rotation."""
        torch.manual_seed(key)
        out = SO2.from_radians(
            torch.rand(
                batch_axes
            )* 2.0 * torch.pi
        )
        assert out.get_batch_axes() == batch_axes
        return out

if __name__ == "__main__":
    # Example usage:
    so2 = SO2.from_radians(torch.tensor(1.2))
    # print(so2)
    # print(so2.as_matrix())
    so2.sample_uniform(0, batch_axes=(2, 3))
    # print(so2.sample_uniform(2))
    # print(so2.normalize())
    # print(so2.log())
    # print(so2.identity())
