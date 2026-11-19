"""Nonlinear kinematics of Asimov-1's parallel ankle linkage, in torch (no Isaac Lab import).

Motors A and B drive crank-rod branches between the shank and the foot. For the virtual ankle joints
``q = (pitch, roll)`` this module gives the motor angles ``q_m = g(q)`` (closed form) and the Jacobian
``J(q) = dq_m/dq`` (implicit differentiation of the rod-length constraints), in the firmware's motor sign
convention. At the zero pose ``J`` reproduces the firmware's linear map
``T0 = [[k_p, -k_r], [-k_p, -k_r]]`` with ``k_p = 2.02`` and ``k_r = 0.80``.

Left leg, knee-link frame (x forward, y left, z up). The right leg mirrors the left as the firmware does:
``g_R(q) = -g_L(-q)`` and ``J_R(q) = J_L(-q)``.

The geometry was estimated from the simulation meshes and the URDF joint origins (not from CAD); it is the same
model as ``analysis/ankle_kinematics/ankle_kin.py`` in the paper repository.
"""

from __future__ import annotations

import math

import torch

# Linear transmission of the firmware (rad of motor per rad of joint).
K_PITCH = 2.02
K_ROLL = 0.80

# --- geometry (m), left leg, knee-link frame ---
_O_P = (0.0, 0.0, -0.29466)  # ankle pitch axis (y), URDF knee -> ankle_pitch origin
_O_R_FROM_P = (-0.0015, 0.0, -0.01)  # ankle roll axis (-x) relative to the pitch axis
_RHO = 0.0405 / K_PITCH  # crank radius
_F = {"A": (0.042, -0.016, 0.0), "B": (0.042, 0.016, 0.0)}  # rod ends on the foot-side bar, roll-link frame
_M = {"A": (0.0405, 0.0, -0.0827), "B": (0.0405, 0.0, -0.1527)}  # crank axes (along x)
_S = {"A": -1.0, "B": 1.0}  # crank direction at theta = 0 (A points -y, B +y)


def _foot_point(k: str, p: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Rod end on the foot (x, y, z) and its derivatives with respect to pitch and roll."""
    fx, fy, fz = _F[k]
    cr, sr = torch.cos(r), torch.sin(r)
    # roll about -x: Rx(-r) @ F
    ry_ = cr * fy + sr * fz
    rz_ = -sr * fy + cr * fz
    dry_dr = -sr * fy + cr * fz
    drz_dr = -cr * fy - sr * fz
    # vector from the pitch axis, in the pitch frame
    vx = _O_R_FROM_P[0] + fx
    vy = _O_R_FROM_P[1] + ry_
    vz = _O_R_FROM_P[2] + rz_
    cp, sp = torch.cos(p), torch.sin(p)
    # Ry(p) @ v
    x = _O_P[0] + cp * vx + sp * vz
    y = _O_P[1] + vy
    z = _O_P[2] - sp * vx + cp * vz
    dx_dp, dy_dp, dz_dp = -sp * vx + cp * vz, torch.zeros_like(p), -cp * vx - sp * vz
    dx_dr, dy_dr, dz_dr = sp * drz_dr, dry_dr, cp * drz_dr
    return x, y, z, (dx_dp, dy_dp, dz_dp), (dx_dr, dy_dr, dz_dr)


def _rod_length(k: str) -> float:
    x, y, z, *_ = _foot_point(k, torch.zeros((), dtype=torch.float64), torch.zeros((), dtype=torch.float64))
    mx, my, mz = _M[k]
    ex, ey, ez = mx, my + _RHO * _S[k], mz
    return math.sqrt(float((ex - x) ** 2 + (ey - y) ** 2 + (ez - z) ** 2))


_L = {k: _rod_length(k) for k in "AB"}


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def _crank(k: str, p: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raw crank angle of branch k and its derivatives d(theta)/d(pitch), d(theta)/d(roll)."""
    x, y, z, dX_dp, dX_dr = _foot_point(k, p, r)
    mx, my, mz = _M[k]
    dx, dy, dz = x - mx, y - my, z - mz
    s = _S[k]
    a, b = s * dy, dz
    kk = (_RHO**2 + dx * dx + dy * dy + dz * dz - _L[k] ** 2) / (2.0 * _RHO)
    radius = torch.sqrt(a * a + b * b)
    phi = torch.atan2(b, a)
    half = torch.acos(torch.clamp(kk / radius, -1.0, 1.0))
    c1, c2 = _wrap(phi + half), _wrap(phi - half)
    theta = torch.where(c1.abs() <= c2.abs(), c1, c2)  # the branch through theta = 0 at the zero pose
    # implicit derivative of f = u(theta).D - K = 0, with u = (0, s cos, sin)
    ux, uy, uz = torch.zeros_like(theta), s * torch.cos(theta), torch.sin(theta)
    df_dtheta = -s * torch.sin(theta) * dy + torch.cos(theta) * dz
    wx, wy, wz = ux - dx / _RHO, uy - dy / _RHO, uz - dz / _RHO
    df_dp = wx * dX_dp[0] + wy * dX_dp[1] + wz * dX_dp[2]
    df_dr = wx * dX_dr[0] + wy * dX_dr[1] + wz * dX_dr[2]
    return theta, -df_dp / df_dtheta, -df_dr / df_dtheta


def _sign() -> tuple[float, float]:
    """Motor signs that make J(0) match the firmware pattern [[+k_p, -k_r], [-k_p, -k_r]]."""
    zero = torch.zeros((), dtype=torch.float64)
    _, a_p, _ = _crank("A", zero, zero)
    _, b_p, _ = _crank("B", zero, zero)
    return float(torch.sign(a_p)), float(-torch.sign(b_p))


_SIGN_A, _SIGN_B = _sign()


def left_motor_angles_and_jacobian(pitch: torch.Tensor, roll: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Left ankle: ``q_m`` with shape (..., 2) and ``J`` with shape (..., 2, 2) for joint angles of shape (...)."""
    ta, ta_p, ta_r = _crank("A", pitch, roll)
    tb, tb_p, tb_r = _crank("B", pitch, roll)
    qm = torch.stack((_SIGN_A * ta, _SIGN_B * tb), dim=-1)
    jac = torch.stack(
        (torch.stack((_SIGN_A * ta_p, _SIGN_A * ta_r), dim=-1), torch.stack((_SIGN_B * tb_p, _SIGN_B * tb_r), dim=-1)),
        dim=-2,
    )
    return qm, jac


def motor_angles_and_jacobian(
    pitch: torch.Tensor, roll: torch.Tensor, right: bool | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Either leg. ``right`` broadcasts against the joint angles; the right leg uses the firmware's mirror rule."""
    right = torch.as_tensor(right, device=pitch.device)
    flip = torch.where(right, -1.0, 1.0).to(pitch.dtype)
    qm, jac = left_motor_angles_and_jacobian(flip * pitch, flip * roll)
    return flip.unsqueeze(-1) * qm, jac


def linear_motor_angles(pitch: torch.Tensor, roll: torch.Tensor) -> torch.Tensor:
    """The firmware's linear map ``T0 q`` (the same for both legs)."""
    return torch.stack((K_PITCH * pitch - K_ROLL * roll, -K_PITCH * pitch - K_ROLL * roll), dim=-1)
