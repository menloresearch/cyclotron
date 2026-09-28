# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Per-episode and per-category metrics, mirroring the get-up acceptance checks (time-to-stand + hold, torque
saturation, weak-motor effort scale) and (in spirit, since ``scripts/getup/evaluate.py`` targets the real Isaac
``getup_state``/contact-sensor fields, which don't exist in this MuJoCo harness) the shape of its metrics JSON: per-category success / time-to-stand / torque-saturation / jitter / impact.
Constants (thresholds) are duplicated from ``scripts/getup/_common.py`` in :mod:`constants`, numerically identical.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from . import constants as C
from .episode import TickRecord


def _tilt_rad(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    # world-frame body z-axis: R(q) @ [0,0,1]
    bz = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    return float(np.arccos(np.clip(bz[2], -1.0, 1.0)))


FOOT_CONTACT_FORCE_N = 1.0
"""Matches ``tasks/getup/mdp/state.py::FOOT_CONTACT_FORCE`` exactly."""


def _standing_conditions(rec: TickRecord) -> dict[str, bool]:
    """Each of the four ``is_standing`` sub-conditions, individually -- lets a caller see *which* one(s) fail at a
    given tick, not just the AND of all four (e.g. which condition breaks when a stand wobbles and briefly loses
    is_standing)."""
    tilt = _tilt_rad(rec.root_quat)
    return {
        "height": rec.root_pos[2] >= C.STANDING_PELVIS_HEIGHT_M,
        "tilt": tilt <= C.STANDING_TILT_RAD,
        "feet_contact": rec.feet_force[0] > FOOT_CONTACT_FORCE_N and rec.feet_force[1] > FOOT_CONTACT_FORCE_N,
        "lin_vel": float(np.linalg.norm(rec.root_lin_vel)) < C.STANDING_MAX_LIN_VEL,
    }


def is_standing_tick(rec: TickRecord) -> bool:
    return all(_standing_conditions(rec).values())


def failed_conditions(rec: TickRecord) -> list[str]:
    """Names of the sub-condition(s) that are False at this tick (empty if standing)."""
    return [k for k, v in _standing_conditions(rec).items() if not v]


@dataclass
class EpisodeMetrics:
    category: str
    control_start_t: float
    episode_len_s: float
    success_1s: bool
    success_g1: bool
    time_to_stand_s: float | None
    max_standing_hold_s: float
    torque_saturation_frac: float  # all-joint pooled fraction of (tick, joint) pairs with tau_hat >= 0.9
    elbow_wrist_saturation_time_s: float  # either arm (existing, kept for continuity)
    action_jitter_rms: float
    peak_non_foot_impact_force_n: float  # raw single-substep peak (can be a rigid-contact solver spike)
    peak_non_foot_impact_force_smoothed5_n: float  # peak of a 5-substep moving-average window
    peak_pelvis_height_m: float
    final_pelvis_height_m: float
    peak_tau_hat_overall: float  # max |torque|/effort_limit over the whole episode, any joint (tau_hat)
    peak_tau_hat_joint: str  # which joint hit that peak
    peak_tau_hat_per_joint: list[float] = field(default_factory=list)  # 23 values, ASIMOV_1_JOINT_NAMES order
    per_joint_saturation_frac: list[float] = field(default_factory=list)  # 23 values: fraction of TICKS tau_hat>=0.9
    left_arm_saturation_time_s: float = 0.0  # left elbow OR left wrist_yaw tau_hat >= 0.9, seconds
    right_arm_saturation_time_s: float = 0.0  # right elbow OR right wrist_yaw tau_hat >= 0.9, seconds
    streak_break_causes: list[str] = field(default_factory=list)  # one "+"-joined entry per mid-episode streak-end


def compute_episode_metrics(
    history: list[TickRecord], category: str, control_start_t: float, policy_dt: float = C.POLICY_DT
) -> EpisodeMetrics:
    if not history:
        raise ValueError("empty episode history")
    standing_flags = [is_standing_tick(r) for r in history]
    ts = [r.t for r in history]

    # longest continuous "standing" streak, and the earliest streak whose START is within the time-to-stand window
    # (G1_TIME_TO_STAND_MAX_S) and whose length reaches G1_HOLD_S. Also record *why* each streak that ends mid-episode (a genuine "lost is_standing"
    # event, as opposed to just running out of episode while still standing) ended -- which of the 4 sub-conditions
    # newly failed at that tick.
    streak_start = None
    streaks: list[tuple[float, float]] = []  # (start_t, end_t)
    streak_break_causes: list[list[str]] = []  # one entry per streak that ended mid-episode (not at episode end)
    for i, (flag, t) in enumerate(zip(standing_flags, ts)):
        if flag and streak_start is None:
            streak_start = t
        elif not flag and streak_start is not None:
            streaks.append((streak_start, t))
            streak_break_causes.append(failed_conditions(history[i]))
            streak_start = None
    if streak_start is not None:
        streaks.append((streak_start, ts[-1]))

    max_hold = max((e - s for s, e in streaks), default=0.0)
    success_1s = any((e - s) >= C.STANDING_HOLD_S for s, e in streaks)
    time_to_stand = None
    success_g1 = False
    for s, e in streaks:
        rel_start = s - control_start_t
        if rel_start <= C.G1_TIME_TO_STAND_MAX_S and (e - s) >= C.G1_HOLD_S:
            success_g1 = True
            time_to_stand = rel_start if time_to_stand is None else min(time_to_stand, rel_start)
    if time_to_stand is None:
        for flag, t in zip(standing_flags, ts):
            if flag:
                time_to_stand = t - control_start_t
                break

    torque = np.stack([r.torque for r in history])  # (T, 23)
    limit = np.stack([r.effort_limit for r in history])
    tau_hat = np.abs(torque) / np.clip(limit, 1e-6, None)  # (T, 23)
    sat_frac = float(np.mean(tau_hat >= C.G2_TAU_HAT_SAT_THRESHOLD))
    peak_tau_hat_per_joint = tau_hat.max(axis=0)  # (23,)
    peak_joint_idx = int(np.argmax(peak_tau_hat_per_joint))

    per_joint_sat_frac = (tau_hat >= C.G2_TAU_HAT_SAT_THRESHOLD).mean(axis=0)  # (23,) fraction of ticks, per joint

    ew_mask = np.array(["elbow" in n or "wrist" in n for n in C.ASIMOV_1_JOINT_NAMES])
    ew_sat = tau_hat[:, ew_mask] >= C.G2_TAU_HAT_SAT_THRESHOLD
    ew_sat_any = ew_sat.any(axis=1)
    ew_sat_time = float(np.sum(ew_sat_any) * policy_dt)

    # per-arm: the elbow/wrist saturation split left vs right -- it can be strongly one-arm-dominant, so pooling
    # both arms into one number hides that asymmetry.
    left_mask = np.array([n.startswith("left_") and ("elbow" in n or "wrist" in n) for n in C.ASIMOV_1_JOINT_NAMES])
    right_mask = np.array([n.startswith("right_") and ("elbow" in n or "wrist" in n) for n in C.ASIMOV_1_JOINT_NAMES])
    left_sat_time = float(np.sum((tau_hat[:, left_mask] >= C.G2_TAU_HAT_SAT_THRESHOLD).any(axis=1)) * policy_dt)
    right_sat_time = float(np.sum((tau_hat[:, right_mask] >= C.G2_TAU_HAT_SAT_THRESHOLD).any(axis=1)) * policy_dt)

    actions = np.stack([r.action_obs for r in history])  # (T, 23)
    if actions.shape[0] >= 4:
        third_diff = np.diff(actions, n=3, axis=0)
        jitter_rms = float(np.sqrt(np.mean(third_diff**2)))
    else:
        jitter_rms = 0.0

    peak_impact = max((r.non_foot_impact_force for r in history), default=0.0)
    peak_impact_smoothed5 = max((r.non_foot_impact_force_smoothed5 for r in history), default=0.0)
    heights = [r.root_pos[2] for r in history]

    return EpisodeMetrics(
        category=category,
        control_start_t=control_start_t,
        episode_len_s=ts[-1] - ts[0],
        success_1s=success_1s,
        success_g1=success_g1,
        time_to_stand_s=time_to_stand,
        max_standing_hold_s=max_hold,
        torque_saturation_frac=sat_frac,
        elbow_wrist_saturation_time_s=ew_sat_time,
        action_jitter_rms=jitter_rms,
        peak_non_foot_impact_force_n=float(peak_impact),
        peak_non_foot_impact_force_smoothed5_n=float(peak_impact_smoothed5),
        peak_pelvis_height_m=float(max(heights)),
        final_pelvis_height_m=float(heights[-1]),
        peak_tau_hat_overall=float(peak_tau_hat_per_joint[peak_joint_idx]),
        peak_tau_hat_joint=C.ASIMOV_1_JOINT_NAMES[peak_joint_idx],
        peak_tau_hat_per_joint=peak_tau_hat_per_joint.tolist(),
        per_joint_saturation_frac=per_joint_sat_frac.tolist(),
        left_arm_saturation_time_s=left_sat_time,
        right_arm_saturation_time_s=right_sat_time,
        streak_break_causes=["+".join(c) for c in streak_break_causes],
    )


@dataclass
class RunSummary:
    mode: str
    physics_dt: float
    effort_scale: float | dict
    per_episode: list[dict] = field(default_factory=list)

    def add(self, m: EpisodeMetrics) -> None:
        self.per_episode.append(asdict(m))

    def per_category(self) -> dict:
        by_cat: dict[str, list[dict]] = {}
        for m in self.per_episode:
            by_cat.setdefault(m["category"], []).append(m)
        out = {}
        for cat, eps in by_cat.items():
            n = len(eps)
            per_joint_stack = np.stack([e["peak_tau_hat_per_joint"] for e in eps])  # (n, 23)
            per_joint_peak = per_joint_stack.max(axis=0)  # (23,) worst episode per joint
            worst_idx = int(np.argmax(per_joint_peak))
            per_joint_frac_stack = np.stack([e["per_joint_saturation_frac"] for e in eps])  # (n, 23)
            per_joint_frac_mean = per_joint_frac_stack.mean(axis=0)  # (23,) mean fraction-of-ticks, across episodes
            out[cat] = {
                "n": n,
                "success_1s_rate": sum(e["success_1s"] for e in eps) / n,
                "success_g1_rate": sum(e["success_g1"] for e in eps) / n,
                "time_to_stand_s_mean": _mean([e["time_to_stand_s"] for e in eps if e["time_to_stand_s"] is not None]),
                "torque_saturation_frac_mean": _mean([e["torque_saturation_frac"] for e in eps]),
                "elbow_wrist_saturation_time_s_max": max((e["elbow_wrist_saturation_time_s"] for e in eps), default=0.0),
                "left_arm_saturation_time_s_max": max((e["left_arm_saturation_time_s"] for e in eps), default=0.0),
                "right_arm_saturation_time_s_max": max((e["right_arm_saturation_time_s"] for e in eps), default=0.0),
                "action_jitter_rms_mean": _mean([e["action_jitter_rms"] for e in eps]),
                "peak_non_foot_impact_force_n_max": max((e["peak_non_foot_impact_force_n"] for e in eps), default=0.0),
                "peak_non_foot_impact_force_smoothed5_n_max": max(
                    (e["peak_non_foot_impact_force_smoothed5_n"] for e in eps), default=0.0
                ),
                "peak_tau_hat_overall_max": float(per_joint_peak[worst_idx]),
                "peak_tau_hat_worst_joint": C.ASIMOV_1_JOINT_NAMES[worst_idx],
                "peak_tau_hat_per_joint": {n_: float(v) for n_, v in zip(C.ASIMOV_1_JOINT_NAMES, per_joint_peak)},
                "per_joint_saturation_frac": {n_: float(v) for n_, v in zip(C.ASIMOV_1_JOINT_NAMES, per_joint_frac_mean)},
                "streak_break_cause_counts": _count_break_causes(eps),
            }
        return out

    def to_json(self, path: str) -> None:
        payload = {
            "mode": self.mode,
            "physics_dt": self.physics_dt,
            "effort_scale": self.effort_scale,
            "overall_success_1s_rate": _mean([e["success_1s"] for e in self.per_episode]),
            "overall_success_g1_rate": _mean([e["success_g1"] for e in self.per_episode]),
            "per_category": self.per_category(),
            "per_episode": self.per_episode,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)


def _mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def _count_break_causes(eps: list[dict]) -> dict:
    """Aggregates ``streak_break_causes`` (one "+"-joined entry per mid-episode "lost is_standing" event) across
    episodes: how many break events total, and how many involved each of the 4 sub-conditions (a break with more
    than one condition failing simultaneously counts toward each -- reported separately as "single_cause_counts"
    for the cleaner signal of what breaks *alone*)."""
    all_events = [c for e in eps for c in e.get("streak_break_causes", [])]
    per_condition = {"height": 0, "tilt": 0, "feet_contact": 0, "lin_vel": 0}
    single_cause = dict(per_condition)
    for event in all_events:
        conditions = event.split("+")
        for c in conditions:
            per_condition[c] += 1
        if len(conditions) == 1:
            single_cause[conditions[0]] += 1
    return {
        "total_break_events": len(all_events),
        "involved_in_break_counts": per_condition,
        "sole_cause_counts": single_cause,
    }
