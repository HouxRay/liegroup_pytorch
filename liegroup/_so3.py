from __future__ import annotations

from typing import Tuple, Type

import torch
import torch.nn as nn
from typing_extensions import override
import math
from . import _base, hints
from .utils import broadcast_leading_axes, get_epsilon, register_lie_group


@register_lie_group(
    matrix_dim=3,
    parameters_dim=4,
    tangent_dim=3,
    space_dim=3,
)
class SO3(_base.SOBase):
    """Special orthogonal group for 3D rotations. Broadcasting rules are the same as
    for numpy.

    Internal parameterization is `(qw, qx, qy, qz)`. Tangent parameterization is
    `(omega_x, omega_y, omega_z)`.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, wxyz: torch.Tensor):
        """
        Args:
            wxyz: Internal parameters. `(w, x, y, z)` quaternion. Shape should be `(*, 4)`.
        """
        #super().__init__(wxyz)
        self.wxyz = wxyz.to(SO3.device)

    @override
    def __repr__(self) -> str:
        wxyz = torch.round(self.wxyz, decimals=5).to(self.device)
        return f"{self.__class__.__name__}(wxyz={wxyz})"

    @staticmethod
    def from_x_radians(theta: hints.Scalar) -> SO3:
        """Generates a x-axis rotation.

        Args:
            angle: X rotation, in radians.

        Returns:
            Output.
        """
        zeros = torch.zeros_like(theta).to(SO3.device)
        return SO3.exp(torch.stack([theta, zeros, zeros], dim=-1))

    @staticmethod
    def from_y_radians(theta: hints.Scalar) -> SO3:
        """Generates a y-axis rotation.

        Args:
            angle: Y rotation, in radians.

        Returns:
            Output.
        """
        zeros = torch.zeros_like(theta).to(SO3.device)
        return SO3.exp(torch.stack([zeros, theta, zeros], dim=-1))

    @staticmethod
    def from_z_radians(theta: hints.Scalar) -> SO3:
        """Generates a z-axis rotation.

        Args:
            angle: Z rotation, in radians.

        Returns:
            Output.
        """
        zeros = torch.zeros_like(theta).to(SO3.device)
        return SO3.exp(torch.stack([zeros, zeros, theta], dim=-1))

    @staticmethod
    def from_rpy_radians(
        roll: hints.Scalar,
        pitch: hints.Scalar,
        yaw: hints.Scalar,
    ) -> SO3:
        """Generates a transform from a set of Euler angles. Uses the ZYX mobile robot
        convention.

        Args:
            roll: X rotation, in radians. Applied first.
            pitch: Y rotation, in radians. Applied second.
            yaw: Z rotation, in radians. Applied last.

        Returns:
            Output.
        """
        return (
            SO3.from_z_radians(yaw)
            @ SO3.from_y_radians(pitch)
            @ SO3.from_x_radians(roll)
        )

    @staticmethod
    def from_quaternion_xyzw(xyzw: hints.Array) -> SO3:
        """Construct a rotation from an `xyzw` quaternion.

        Note that `wxyz` quaternions can be constructed using the default dataclass
        constructor.

        Args:
            xyzw: xyzw quaternion. Shape should be (*, 4).

        Returns:
            Output.
        """
        assert xyzw.shape[-1:] == (4,)
        return SO3(torch.roll(xyzw, shifts=1, dims=-1))

    def as_quaternion_xyzw(self) -> torch.Tensor:
        """Grab parameters as xyzw quaternion."""
        return torch.roll(self.wxyz, shifts=-1, dims=-1).to(SO3.device)

    def as_rpy_radians(self) -> hints.RollPitchYaw:
        """Computes roll, pitch, and yaw angles. Uses the ZYX mobile robot convention.

        Returns:
            Named tuple containing Euler angles in radians.
        """
        return hints.RollPitchYaw(
            roll=self.compute_roll_radians(),
            pitch=self.compute_pitch_radians(),
            yaw=self.compute_yaw_radians(),
        )

    def compute_roll_radians(self) -> torch.Tensor:
        """Compute roll angle. Uses the ZYX mobile robot convention.

        Returns:
            Euler angle in radians.
        """
        q0, q1, q2, q3 = torch.unbind(self.wxyz, dim=-1)
        return torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))

    def compute_pitch_radians(self) -> torch.Tensor:
        """Compute pitch angle. Uses the ZYX mobile robot convention.

        Returns:
            Euler angle in radians.
        """
        q0, q1, q2, q3 = torch.unbind(self.wxyz, dim=-1)
        return torch.asin(2 * (q0 * q2 - q3 * q1))

    def compute_yaw_radians(self) -> torch.Tensor:
        """Compute yaw angle. Uses the ZYX mobile robot convention.

        Returns:
            Euler angle in radians.
        """
        q0, q1, q2, q3 = torch.unbind(self.wxyz, dim=-1)
        return torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))

    # Factory.

    @classmethod
    @override
    def identity(cls, batch_axes: Tuple[int, ...] = ()) -> SO3:
        return SO3(
            wxyz=torch.broadcast_to(torch.tensor([1.0, 0.0, 0.0, 0.0]), (*batch_axes, 4)).to(cls.device)
        )

    @classmethod
    @override
    def from_matrix(cls, matrix: hints.Array) -> SO3:
        assert matrix.shape[-2:] == (3, 3)

        # Modified from:
        # > "Converting a Rotation Matrix to a Quaternion" from Mike Day
        # > https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf

        def case0(m):
            t = 1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2]
            q = torch.stack(
                [
                    m[..., 2, 1] - m[..., 1, 2],
                    t,
                    m[..., 1, 0] + m[..., 0, 1],
                    m[..., 0, 2] + m[..., 2, 0],
                ],
                dim=-1,
            ).to(SO3.device)
            return t, q

        def case1(m):
            t = 1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2]
            q = torch.stack(
                [
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] + m[..., 0, 1],
                    t,
                    m[..., 2, 1] + m[..., 1, 2],
                ],
                dim=-1,
            ).to(SO3.device)
            return t, q

        def case2(m):
            t = 1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]
            q = torch.stack(
                [
                    m[..., 1, 0] - m[..., 0, 1],
                    m[..., 0, 2] + m[..., 2, 0],
                    m[..., 2, 1] + m[..., 1, 2],
                    t,
                ],
                dim=-1,
            ).to(SO3.device)
            return t, q

        def case3(m):
            t = 1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
            q = torch.stack(
                [
                    t,
                    m[..., 2, 1] - m[..., 1, 2],
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] - m[..., 0, 1],
                ],
                dim=-1,
            ).to(SO3.device)
            return t, q

        # Compute four cases, then pick the most precise one.
        case0_t, case0_q = case0(matrix)
        case1_t, case1_q = case1(matrix)
        case2_t, case2_q = case2(matrix)
        case3_t, case3_q = case3(matrix)

        cond0 = matrix[..., 2, 2] < 0
        cond1 = matrix[..., 0, 0] > matrix[..., 1, 1]
        cond2 = matrix[..., 0, 0] < -matrix[..., 1, 1]

        t = torch.where(
            cond0,
            torch.where(cond1, case0_t, case1_t),
            torch.where(cond2, case2_t, case3_t),
        ).to(SO3.device)
        q = torch.where(
            cond0.unsqueeze(-1),
            torch.where(cond1.unsqueeze(-1), case0_q, case1_q),
            torch.where(cond2.unsqueeze(-1), case2_q, case3_q),
        ).to(SO3.device)

        return SO3(wxyz=q * 0.5 / torch.sqrt(t.unsqueeze(-1)))

    # Accessors.

    @override
    def as_matrix(self) -> torch.Tensor:
        norm_sq = torch.sum(torch.square(self.wxyz), dim=-1, keepdim=True).to(SO3.device)
        q = self.wxyz * torch.sqrt(2.0 / norm_sq).to(SO3.device)  # (*, 4)
        q_outer = torch.einsum("...i,...j->...ij", q, q).to(SO3.device)  # (*, 4, 4)
        return torch.stack(
            [
                1.0 - q_outer[..., 2, 2] - q_outer[..., 3, 3],
                q_outer[..., 1, 2] - q_outer[..., 3, 0],
                q_outer[..., 1, 3] + q_outer[..., 2, 0],
                q_outer[..., 1, 2] + q_outer[..., 3, 0],
                1.0 - q_outer[..., 1, 1] - q_outer[..., 3, 3],
                q_outer[..., 2, 3] - q_outer[..., 1, 0],
                q_outer[..., 1, 3] - q_outer[..., 2, 0],
                q_outer[..., 2, 3] + q_outer[..., 1, 0],
                1.0 - q_outer[..., 1, 1] - q_outer[..., 2, 2],
            ],
            dim=-1,
        ).reshape(*q.shape[:-1], 3, 3)

    @override
    def parameters(self) -> torch.Tensor:
        return self.wxyz

    # Operations.

    @override
    def apply(self, target: hints.Array) -> torch.Tensor:
        assert target.shape[-1:] == (3,)
        self, target = broadcast_leading_axes((self, target))

        # Compute using quaternion multiplys.
        padded_target = torch.cat(
            [torch.zeros((*self.get_batch_axes(), 1), device=target.device), target], dim=-1
        )
        return (self @ SO3(wxyz=padded_target) @ self.inverse()).wxyz[..., 1:]

    @override
    def multiply(self, other: SO3) -> SO3:
        w0, x0, y0, z0 = torch.unbind(self.wxyz, dim=-1)
        w1, x1, y1, z1 = torch.unbind(other.wxyz, dim=-1)
        return SO3(
            wxyz=torch.stack(
                [
                    -x0 * x1 - y0 * y1 - z0 * z1 + w0 * w1,
                    x0 * w1 + y0 * z1 - z0 * y1 + w0 * x1,
                    -x0 * z1 + y0 * w1 + z0 * x1 + w0 * y1,
                    x0 * y1 - y0 * x1 + z0 * w1 + w0 * z1,
                ],
                dim=-1,
            )
        )

    @classmethod
    @override
    def exp(cls: Type['SO3'], tangent: torch.Tensor) -> 'SO3':
        assert tangent.shape[-1] == 3
        tangent = tangent.to(SO3.device)
        
        theta_squared = torch.sum(torch.square(tangent), dim=-1).to(SO3.device)
        theta_pow_4 = theta_squared * theta_squared
        use_taylor = theta_squared < get_epsilon(tangent.dtype)

        safe_theta = torch.sqrt(
            torch.where(
                use_taylor,
                torch.ones_like(theta_squared),
                theta_squared,
            )
        ).to(SO3.device)
        safe_half_theta = 0.5 * safe_theta

        real_factor = torch.where(
            use_taylor,
            1.0 - theta_squared / 8.0 + theta_pow_4 / 384.0,
            torch.cos(safe_half_theta),
        ).to(SO3.device)

        imaginary_factor = torch.where(
            use_taylor,
            0.5 - theta_squared / 48.0 + theta_pow_4 / 3840.0,
            torch.sin(safe_half_theta) / safe_theta,
        ).to(SO3.device)

        wxyz = torch.cat(
            [
                real_factor.unsqueeze(-1),
                imaginary_factor.unsqueeze(-1) * tangent,
            ],
            dim=-1,
        ).to(SO3.device)

        return cls(wxyz)

    @override
    def log(self) -> torch.Tensor:
        w = self.wxyz[..., 0]
        norm_sq = torch.sum(torch.square(self.wxyz[..., 1:]), dim=-1).to(SO3.device)
        use_taylor = norm_sq < get_epsilon(norm_sq.dtype)

        norm_safe = torch.sqrt(
            torch.where(
                use_taylor,
                torch.ones_like(norm_sq),
                norm_sq,
            )
        ).to(SO3.device)
        w_safe = torch.where(use_taylor, w, torch.ones_like(w)).to(SO3.device)
        atan_n_over_w = torch.atan2(
            torch.where(w < 0, -norm_safe, norm_safe),
            torch.abs(w),
        ).to(SO3.device)
        atan_factor = torch.where(
            use_taylor,
            2.0 / w_safe - 2.0 / 3.0 * norm_sq / w_safe**3,
            torch.where(
                torch.abs(w) < get_epsilon(w.dtype),
                torch.where(w > 0, torch.ones_like(w), -torch.ones_like(w)) * math.pi / norm_safe,
                2.0 * atan_n_over_w / norm_safe,
            ),
        ).to(SO3.device)

        return atan_factor.unsqueeze(-1) * self.wxyz[..., 1:]
    @override
    def adjoint(self) -> torch.Tensor:
        return self.as_matrix()

    @override
    def inverse(self) -> SO3:
        return SO3(wxyz=self.wxyz * torch.tensor([1, -1, -1, -1], device=self.wxyz.device))

    @override
    def normalize(self) -> SO3:
        return SO3(wxyz=self.wxyz / torch.linalg.norm(self.wxyz, dim=-1, keepdim=True))

    @classmethod
    @override
    def sample_uniform(cls, key: torch.Tensor, batch_axes: Tuple[int, ...] = ()) -> SO3:
            # 创建一个 Generator 对象
        generator = torch.Generator()
        generator.manual_seed(key)  # 使用提供的 key 来设置随机种子

        # 生成随机四元数
        wxyz = torch.randn((*batch_axes, 4), generator=generator).to(SO3.device)
  
        return cls(wxyz=wxyz).normalize()
    @classmethod
    def sample_unit(cls, seed: int, batch_axes: Tuple[int, ...] = ()) -> SO3:
        generator = torch.Generator()
        generator.manual_seed(seed)  # 使用提供的 key 来设置随机种子
        tan = torch.randn((*batch_axes, 3), generator=generator).to(SO3.device)
        return cls.exp(tan).log()

        

    @override
    def get_batch_axes(self) -> Tuple[int, ...]:
        return self.parameters().shape[:-1]