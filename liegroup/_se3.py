from __future__ import annotations

from typing import Tuple, cast

import torch
import torch.nn as nn
from typing_extensions import override

from . import _base, hints
from ._so3 import SO3
from .utils import broadcast_leading_axes, get_epsilon, register_lie_group


def _skew(omega: hints.Array) -> torch.Tensor:
    """Returns the skew-symmetric form of a length-3 vector."""

    wx, wy, wz = omega.unbind(-1)
    zeros = torch.zeros_like(wx).to(device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return torch.stack(
        [zeros, -wz, wy, wz, zeros, -wx, -wy, wx, zeros],
        dim=-1,
    ).reshape(*omega.shape[:-1], 3, 3)


@register_lie_group(
    matrix_dim=4,
    parameters_dim=7,
    tangent_dim=6,
    space_dim=3,
)
class SE3(_base.SEBase[SO3]):
    """Special Euclidean group for proper rigid transforms in 3D. Broadcasting
    rules are the same as for PyTorch.

    Internal parameterization is `(qw, qx, qy, qz, x, y, z)`. Tangent parameterization
    is `(vx, vy, vz, omega_x, omega_y, omega_z)`.
    """

    # SE3-specific.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wxyz_xyz: torch.Tensor
    """Internal parameters. wxyz quaternion followed by xyz translation. Shape should be `(*, 7)`."""

    def __init__(self, wxyz_xyz: torch.Tensor):
        #super().__init__()
        self.wxyz_xyz = wxyz_xyz.to(self.device)

    @override
    def __repr__(self) -> str:
        quat = torch.round(self.wxyz_xyz[..., :4], decimals=5).to(self.device)
        trans = torch.round(self.wxyz_xyz[..., 4:], decimals=5).to(self.device)
        return f"{self.__class__.__name__}(wxyz={quat}, xyz={trans})"

    # SE-specific.

    @classmethod
    @override
    def from_rotation_and_translation(
        cls,
        rotation: SO3,
        translation: hints.Array,
    ) -> SE3:
        assert translation.shape[-1:] == (3,)
        rotation, translation = broadcast_leading_axes((rotation, translation))
        return SE3(wxyz_xyz=torch.cat([rotation.wxyz, translation], dim=-1))

    @override
    def rotation(self) -> SO3:
        return SO3(wxyz=self.wxyz_xyz[..., :4])

    @override
    def translation(self) -> torch.Tensor:
        return self.wxyz_xyz[..., 4:]

    # Factory.

    @classmethod
    @override
    def identity(cls, batch_axes: Tuple[int, ...] = ()) -> SE3:
        return SE3(
            wxyz_xyz=torch.broadcast_to(
                torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), (*batch_axes, 7)
            )
        )

    @classmethod
    @override
    def from_matrix(cls, matrix: hints.Array) -> SE3:
        assert matrix.shape[-2:] == (4, 4)
        # Currently assumes bottom row is [0, 0, 0, 1].
        return SE3.from_rotation_and_translation(
            rotation=SO3.from_matrix(matrix[..., :3, :3]),
            translation=matrix[..., :3, 3],
        )

    # Accessors.

    @override
    def as_matrix(self) -> torch.Tensor:
        matrix = torch.zeros(*self.get_batch_axes(), 4, 4, device=self.wxyz_xyz.device)
        matrix[..., :3, :3] = self.rotation().as_matrix()
        matrix[..., :3, 3] = self.translation()
        matrix[..., 3, 3] = 1.0
        return matrix

    @override
    def parameters(self) -> torch.Tensor:
        return self.wxyz_xyz

    # Operations.

    @classmethod
    @override
    def exp(cls, tangent: hints.Array) -> SE3:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/se3.hpp#L761

        # (x, y, z, omega_x, omega_y, omega_z)
        assert tangent.shape[-1:] == (6,)

        tangent = tangent.to(SO3.device)
        rotation = SO3.exp(tangent[..., 3:])

        theta_squared = torch.sum(torch.square(tangent[..., 3:]).to(SE3.device), dim=-1)
        use_taylor = theta_squared < get_epsilon(theta_squared.dtype)

        # Shim to avoid NaNs in torch.where branches, which cause failures for
        # reverse-mode AD.
        theta_squared_safe = cast(
            torch.Tensor,
            torch.where(
                use_taylor,
                torch.ones_like(theta_squared),  # Any non-zero value should do here.
                theta_squared,
            ),
        ).to(SE3.device)
        del theta_squared
        theta_safe = torch.sqrt(theta_squared_safe).to(SE3.device)

        skew_omega = _skew(tangent[..., 3:])
        V = torch.where(
            use_taylor[..., None, None],
            rotation.as_matrix(),
            (
                torch.eye(3, dtype=tangent.dtype, device=tangent.device)
                + ((1.0 - torch.cos(theta_safe)) / (theta_squared_safe))[..., None, None]
                * skew_omega
                + (
                    (theta_safe - torch.sin(theta_safe))
                    / (theta_squared_safe * theta_safe)
                )[..., None, None]
                * torch.matmul(skew_omega, skew_omega)
            ),
        ).to(SE3.device)

        return SE3.from_rotation_and_translation(
            rotation=rotation,
            translation=torch.einsum("...ij,...j->...i", V, tangent[..., :3]),
        )
    @classmethod
    def exp_matrix(cls, tangent: torch.Tensor) -> torch.Tensor:

        assert tangent.shape[-1] == 6
        
        tangent = tangent.to(SO3.device)
        # 将旋转部分和平移部分分开
        translation = tangent[..., :3]
        omega = tangent[..., 3:]

        # 计算旋转矩阵
        theta_squared = torch.sum(omega ** 2, dim=-1, keepdim=True).to(SE3.device)
        theta = torch.sqrt(theta_squared).to(SE3.device)
        axis = omega / (theta + 1e-7)  # 添加一个小的常数以避免除以零
        cos_theta = torch.cos(theta)[..., None].to(SE3.device)
        sin_theta = torch.sin(theta)[..., None].to(SE3.device)

        rotation_matrix = cos_theta * torch.eye(3, dtype=tangent.dtype, device=tangent.device) + \
                        (1 - cos_theta) * torch.einsum('...i,...j->...ij', axis, axis) + \
                        sin_theta * _skew(axis).to(SE3.device)

        # 计算 V 矩阵
        V = rotation_matrix + \
            ((1 - cos_theta) / (theta_squared + 1e-7))[..., None] * _skew(omega) + \
            ((theta - sin_theta) / (theta_squared * theta + 1e-7))[..., None] * torch.matmul(_skew(omega), _skew(omega))

        # 计算平移向量
        translation_transformed = torch.matmul(V, translation[..., None])[..., 0].to(SE3.device)

        # 构造 SE(3) 矩阵
        se3_matrix = torch.eye(4, dtype=tangent.dtype, device=tangent.device).repeat(*tangent.shape[:-1], 1, 1).to(SE3.device)
        se3_matrix[..., :3, :3] = rotation_matrix
        se3_matrix[..., :3, 3] = translation_transformed.squeeze(-1)  # 移除最后一个维度

        return se3_matrix


    @override
    def log(self) -> torch.Tensor:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/se3.hpp#L223
        omega = self.rotation().log()
        theta_squared = torch.sum(torch.square(omega), dim=-1).to(SE3.device)
        use_taylor = theta_squared < get_epsilon(theta_squared.dtype)

        skew_omega = _skew(omega)

        # Shim to avoid NaNs in torch.where branches, which cause failures for
        # reverse-mode AD.
        theta_squared_safe = torch.where(
            use_taylor,
            torch.ones_like(theta_squared),  # Any non-zero value should do here.
            theta_squared,
        ).to(SE3.device)
        del theta_squared
        theta_safe = torch.sqrt(theta_squared_safe).to(SE3.device)
        half_theta_safe = theta_safe / 2.0

        V_inv = torch.where(
            use_taylor[..., None, None],
            torch.eye(3, dtype=omega.dtype, device=omega.device)
            - 0.5 * skew_omega
            + torch.einsum("...ij,...jk->...ik", skew_omega, skew_omega) / 12.0,
            (
                torch.eye(3, dtype=omega.dtype, device=omega.device)
                - 0.5 * skew_omega
                + (
                    (
                        1.0
                        - theta_safe
                        * torch.cos(half_theta_safe)
                        / (2.0 * torch.sin(half_theta_safe))
                    )
                    / theta_squared_safe
                )[..., None, None]
                * torch.einsum("...ij,...jk->...ik", skew_omega, skew_omega)
            ),
        ).to(SE3.device)
        return torch.cat([torch.einsum("...ij,...j->...i", V_inv, self.translation()), omega], dim=-1)

    @override
    def adjoint(self) -> torch.Tensor:
        R = self.rotation().as_matrix()
        return torch.cat(
            [
                torch.cat(
                    [R, torch.matmul(_skew(self.translation()), R)],
                    dim=-1,
                ),
                torch.cat(
                    [torch.zeros(*self.get_batch_axes(), 3, 3).to(SE3.device), R], dim=-1
                ),
            ],
            dim=-2,
        )

    @classmethod
    @override
    def sample_uniform(
        cls, key: torch.Tensor, batch_axes: Tuple[int, ...] = ()
    ) -> SE3:
        return SE3.from_rotation_and_translation(
            rotation=SO3.sample_uniform(key, batch_axes=batch_axes),
            translation=torch.rand(*batch_axes, 3, device=SE3.device) * 2 - 1,
        )
    
    @classmethod
    def sample_unit(cls, key: int, batch_axes: Tuple[int, ...] = (), device: str = 'cpu') -> SE3:

        dx_rot = SO3.sample_unit(key, batch_axes=batch_axes)
        dx_trans = torch.normal(mean=0, std=1, size=(*batch_axes, 3), generator=torch.Generator().manual_seed(key))

        dx = torch.cat([dx_trans, dx_rot], dim=-1)
        dx.to(device)

        return SE3.exp(dx)
    
    def to_device(self, device: str) -> SE3:
        self.wxyz_xyz.to(device)
        return self
    
      

## add test codes 

if __name__ == '__main__':
    R = SO3.sample_uniform(232, (1,))
    t = torch.tensor([1, 2, 3], dtype=torch.float32)

    print("R as matrix:\n", R.as_matrix())
    print("shape of R as matrix:\n", R.as_matrix().shape)
    SO3.exp(R.log())
    print("R as quaternion:\n", R.wxyz)
    print("R as log:\n", R.log())

    T = SE3.from_rotation_and_translation(rotation=R, translation=t)
    print("T as matrix:\n", T.as_matrix())
    print("shape of T as matrix:\n", T.as_matrix().shape)

    log_sample = T.log()
    print("log_sample:\n", log_sample)

    T_reconstructed = SE3.exp(log_sample)
    print("T_reconstructed as matrix:\n", T_reconstructed.as_matrix())
