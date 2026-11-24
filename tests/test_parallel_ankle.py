"""The parallel-ankle linkage model and the actuator that simulates the robot's motor-side PD through it."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_KINEMATICS = (
    Path(__file__).resolve().parents[1] / "source/cyclotron/cyclotron/assets/robots/asimov_1_ankle_kinematics.py"
)


def _kinematics():
    # Loaded by path: importing the cyclotron.assets.robots package needs Isaac Lab, this module does not.
    spec = importlib.util.spec_from_file_location("asimov_1_ankle_kinematics", _KINEMATICS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


K = _kinematics()
F64 = torch.float64


def _grid():
    p, r = torch.meshgrid(
        torch.linspace(-0.35, 0.35, 15, dtype=F64), torch.linspace(-0.09, 0.09, 7, dtype=F64), indexing="ij"
    )
    return p.reshape(-1), r.reshape(-1)


def test_jacobian_at_zero_is_the_firmware_linear_map():
    zero = torch.zeros(1, dtype=F64)
    qm, jac = K.left_motor_angles_and_jacobian(zero, zero)
    assert torch.allclose(qm, torch.zeros(1, 2, dtype=F64), atol=1e-12)
    t0 = torch.tensor([[K.K_PITCH, -K.K_ROLL], [-K.K_PITCH, -K.K_ROLL]], dtype=F64)
    assert torch.allclose(jac[0], t0, atol=5e-3)


def test_jacobian_matches_finite_differences():
    p, r = _grid()
    _, jac = K.left_motor_angles_and_jacobian(p, r)
    h = 1e-6
    dp = (K.left_motor_angles_and_jacobian(p + h, r)[0] - K.left_motor_angles_and_jacobian(p - h, r)[0]) / (2 * h)
    dr = (K.left_motor_angles_and_jacobian(p, r + h)[0] - K.left_motor_angles_and_jacobian(p, r - h)[0]) / (2 * h)
    assert torch.allclose(jac, torch.stack((dp, dr), dim=-1), atol=1e-6)


def test_right_leg_follows_the_firmware_mirror_rule():
    p, r = _grid()
    qm_r, jac_r = K.motor_angles_and_jacobian(p, r, True)
    qm_l, jac_l = K.left_motor_angles_and_jacobian(-p, -r)
    assert torch.allclose(qm_r, -qm_l)
    assert torch.allclose(jac_r, jac_l)


def test_equal_motor_gains_stay_decoupled_and_reproduce_the_trained_gains_at_zero():
    k = 27.5
    zero = torch.zeros(1, dtype=F64)
    _, jac = K.left_motor_angles_and_jacobian(zero, zero)
    kj = k * jac[0].T @ jac[0]
    assert kj[0, 0].item() == pytest.approx(224.42, rel=2e-3)
    assert kj[1, 1].item() == pytest.approx(35.20, rel=1e-2)
    # Over the walking range (pitch about the -0.3 rad stance, small roll) the coupling stays small; it peaks at about
    # 6 % at the -0.35 rad pitch limit.
    p = torch.linspace(-0.35, -0.05, 13, dtype=F64)
    r = torch.full_like(p, 0.03)
    _, jac = K.left_motor_angles_and_jacobian(p, r)
    kj = k * jac.transpose(-1, -2) @ jac
    coupling = kj[:, 0, 1].abs() / torch.sqrt(kj[:, 0, 0] * kj[:, 1, 1])
    assert coupling.max().item() < 0.08


def test_float32_matches_float64():
    p, r = _grid()
    qm64, jac64 = K.left_motor_angles_and_jacobian(p, r)
    qm32, jac32 = K.left_motor_angles_and_jacobian(p.float(), r.float())
    assert torch.allclose(qm32.double(), qm64, atol=1e-5)
    assert torch.allclose(jac32.double(), jac64, atol=1e-4)


def test_linear_map_is_t0():
    p, r = _grid()
    qm = K.linear_motor_angles(p, r)
    assert torch.allclose(qm[:, 0], K.K_PITCH * p - K.K_ROLL * r)
    assert torch.allclose(qm[:, 1], -K.K_PITCH * p - K.K_ROLL * r)


def test_motor_pd_on_joint_errors_equals_pose_dependent_joint_gains():
    # The actuator's torque, J^T (K_m J e + D_m J de), is a joint-space PD with gains J^T K_m J and J^T D_m J.
    p, r = _grid()
    _, jac = K.left_motor_angles_and_jacobian(p, r)
    km = torch.tensor([27.5, 30.0], dtype=F64)
    dm = torch.tensor([1.25, 1.1], dtype=F64)
    e = torch.stack((0.05 - p, 0.01 - r), dim=-1).unsqueeze(-1)
    de = torch.full_like(e, -0.3)
    jt = jac.transpose(-1, -2)
    tau = jt @ (km.unsqueeze(-1) * (jac @ e) + dm.unsqueeze(-1) * (jac @ de))
    kj = jt @ torch.diag(km) @ jac
    dj = jt @ torch.diag(dm) @ jac
    assert torch.allclose(tau, kj @ e + dj @ de)
