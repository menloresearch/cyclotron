# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Equivalence tests for the sync-free actuator delay (no GPU->host syncs) against Isaac Lab's DelayBuffer / DelayedPDActuator.

Isaac Lab's actuator/buffer modules import ``pxr``, so a headless Kit app is started at import when ``pxr`` is missing.
Run with Isaac Sim installed (the file runs pytest in-process and then exits hard, since ``SimulationApp.close()`` can hang):

    python tests/getup/test_getup_actuators.py

Tests run on CPU and, when available, on CUDA.
"""

from __future__ import annotations

import os
import sys

try:
    import pxr  # noqa: F401
except ImportError:
    from isaaclab.app import AppLauncher

    _APP = AppLauncher(headless=True).app

import pytest  # noqa: E402
import torch  # noqa: E402

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg  # noqa: E402
from isaaclab.utils.buffers import DelayBuffer  # noqa: E402
from isaaclab.utils.types import ArticulationActions  # noqa: E402

from isaac_asimov.assets.robots.getup_actuators import (  # noqa: E402
    DelayedPDLimpableActuator,
    DelayedPDLimpableActuatorCfg,
    SyncFreeDelayBuffer,
    ankle_joint_torques,
    ankle_motor_torques,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _random_ids(n, gen, device):
    mask = torch.rand(n, generator=gen) < 0.3
    return torch.nonzero(mask).flatten().to(device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("max_delay", [0, 1, 5])
def test_ring_buffer_matches_isaaclab_delay_buffer(device, max_delay):
    """Random data, random per-env lags, random partial resets (with new lags): bit-identical outputs."""
    gen = torch.Generator().manual_seed(1234 + max_delay)
    n, j, steps = 64, 7, 300
    ref = DelayBuffer(max_delay, n, device=device)
    new = SyncFreeDelayBuffer(max_delay, n, j, device)
    # initial state: lag 0 everywhere in both (no reset yet), exactly like a freshly built DelayedPDActuator
    for t in range(steps):
        if t % 17 == 3:
            ids = _random_ids(n, gen, device)
            lags = torch.randint(0, max_delay + 1, (len(ids),), generator=gen).to(device)
            ref.set_time_lag(lags.int(), ids)
            ref.reset(ids)
            new.reset(ids, lags.long().unsqueeze(1).expand(-1, j))
        if t == 150:  # full reset
            lags = torch.randint(0, max_delay + 1, (n,), generator=gen).to(device)
            ref.set_time_lag(lags.int())
            ref.reset()
            new.reset(slice(None), lags.long().unsqueeze(1).expand(-1, j))
        x = torch.randn(n, j, generator=gen).to(device)
        a = ref.compute(x)
        b = new.compute(x)
        assert torch.equal(a, b), f"step {t}: max diff {(a - b).abs().max()}"


@pytest.mark.parametrize("device", DEVICES)
def test_per_group_lags_match_one_buffer_per_group(device):
    """Per-(env, joint) lags == one Isaac Lab DelayBuffer per joint group, each with its own per-env lag."""
    gen = torch.Generator().manual_seed(7)
    n, groups, steps, max_delay = 32, [[0, 1, 2], [3], [4, 5]], 200, 5
    j = sum(len(g) for g in groups)
    refs = [DelayBuffer(max_delay, n, device=device) for _ in groups]
    new = SyncFreeDelayBuffer(max_delay, n, j, device)
    group_of_joint = torch.tensor([gi for gi, g in enumerate(groups) for _ in g], device=device)
    for t in range(steps):
        if t % 13 == 0:
            ids = _random_ids(n, gen, device) if t else torch.arange(n, device=device)
            lag_g = torch.randint(0, max_delay + 1, (len(ids), len(groups)), generator=gen).to(device)
            for gi, r in enumerate(refs):
                r.set_time_lag(lag_g[:, gi].int(), ids)
                r.reset(ids)
            new.reset(ids, lag_g[:, group_of_joint].long())
        x = torch.randn(n, j, generator=gen).to(device)
        out = new.compute(x)
        for gi, (g, r) in enumerate(zip(groups, refs)):
            assert torch.equal(out[:, g], r.compute(x[:, g])), f"step {t} group {gi}"


def _cfg_pair(joint_expr, min_delay=0, max_delay=5):
    common = dict(
        joint_names_expr=joint_expr,
        stiffness={"a.*": 150.0, "b.*": 40.0},
        damping={"a.*": 5.0, "b.*": 2.0},
        effort_limit={"a.*": 45.0, "b.*": 12.0},
        armature=0.03,
        friction=0.4,
        min_delay=min_delay,
        max_delay=max_delay,
    )
    return DelayedPDActuatorCfg(**common), DelayedPDLimpableActuatorCfg(**common)


def _make(cfg, names, n, device):
    return cfg.class_type(cfg, joint_names=names, joint_ids=slice(None), num_envs=n, device=device)


@pytest.mark.parametrize("device", DEVICES)
def test_actuator_matches_isaaclab_delayed_pd(device):
    """Full actuator: same torques as Isaac Lab DelayedPDActuator (zero velocity/effort targets, gain_scale 1)."""
    gen = torch.Generator().manual_seed(3)
    names = ["a0", "a1", "b0", "b1", "b2"]
    n, steps = 16, 200
    ref_cfg, new_cfg = _cfg_pair([".*"])
    ref = _make(ref_cfg, names, n, device)
    new = _make(new_cfg, names, n, device)
    assert isinstance(new, DelayedPDLimpableActuator) and isinstance(ref, DelayedPDActuator)
    zeros = torch.zeros(n, len(names), device=device)
    for t in range(steps):
        if t % 25 == 0:
            ids = torch.arange(n, device=device) if t == 0 else _random_ids(n, gen, device)
            ref.reset(ids)
            new.reset(ids)
            # force identical lags (the two classes draw from the RNG differently)
            lag = ref.positions_delay_buffer.time_lags[ids].long()
            new.positions_delay_buffer.lags[ids] = lag.unsqueeze(1).expand(-1, len(names))
        q_des = torch.randn(n, len(names), generator=gen).to(device)
        q = torch.randn(n, len(names), generator=gen).to(device)
        qd = torch.randn(n, len(names), generator=gen).to(device)
        a = ref.compute(ArticulationActions(q_des.clone(), zeros.clone(), zeros.clone()), q, qd)
        b = new.compute(ArticulationActions(q_des.clone(), zeros.clone(), zeros.clone()), q, qd)
        assert torch.equal(a.joint_efforts, b.joint_efforts), f"step {t}"
        assert torch.equal(ref.computed_effort, new.computed_effort)


@pytest.mark.parametrize("device", DEVICES)
def test_delay_groups_and_limp(device):
    """delay_groups gives one lag per (env, group) in [min, max]; gain_scale 0 gives -0.5*qd exactly; reset -> 1."""
    names = ["a0", "a1", "b0", "b1", "b2"]
    n = 256
    _, cfg = _cfg_pair([".*"], min_delay=1, max_delay=4)
    cfg = cfg.replace(delay_groups=[["a.*"], ["b.*"]])
    act = _make(cfg, names, n, device)
    act.reset(None)
    lags = act.positions_delay_buffer.lags
    assert int(lags.min()) >= 1 and int(lags.max()) <= 4
    assert torch.all(lags[:, 0] == lags[:, 1]) and torch.all(lags[:, 2] == lags[:, 4])
    assert bool((lags[:, 0] != lags[:, 2]).any())  # groups are independent
    act.gain_scale[:] = 0.0
    qd = torch.randn(n, len(names), device=device)
    z = torch.zeros(n, len(names), device=device)
    out = act.compute(ArticulationActions(torch.randn(n, len(names), device=device), z.clone(), z.clone()), z, qd)
    assert torch.allclose(out.joint_efforts, torch.clamp(-0.5 * qd, -12.0, 12.0))
    act.reset(torch.arange(0, n, 2, device=device))
    assert torch.all(act.gain_scale[0::2] == 1.0) and torch.all(act.gain_scale[1::2] == 0.0)


def test_ankle_transform_round_trip():
    tp, tr = torch.randn(1000) * 40, torch.randn(1000) * 17
    ta, tb = ankle_motor_torques(tp, tr)
    p2, r2 = ankle_joint_torques(ta, tb)
    assert torch.allclose(tp, p2, atol=1e-4) and torch.allclose(tr, r2, atol=1e-4)
    # virtual work: tau_q . qd == tau_m . md for the position map m = J q
    qd = torch.randn(1000, 2)
    ma, mb = 2.02 * qd[:, 0] - 0.80 * qd[:, 1], -2.02 * qd[:, 0] - 0.80 * qd[:, 1]
    assert torch.allclose(tp * qd[:, 0] + tr * qd[:, 1], ta * ma + tb * mb, atol=1e-3)


if __name__ == "__main__":
    code = pytest.main([__file__, "-q", "-p", "no:cacheprovider"])
    sys.stdout.flush()
    os._exit(int(code))
