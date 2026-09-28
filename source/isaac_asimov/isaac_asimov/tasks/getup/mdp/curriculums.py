# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# `assist_decay` follows NVIDIA WBC-AGILE `adaptive_force_decay` and `reward_weight_ramp` follows AGILE
# `update_reward_weight_step` (both `agile/rl_env/mdp/curriculums/task_curriculum.py`, Copyright (c) 2025 NVIDIA
# CORPORATION & AFFILIATES, Apache-2.0, http://www.apache.org/licenses/LICENSE-2.0). Rewritten for the shared get-up
# state; iteration-based instead of step-based schedules.
"""Get-up curricula.

Curriculum terms run first in ``_reset_idx`` (before reset events and manager resets), so they see the final state of
the episodes that just ended. Iterations are derived from ``env.common_step_counter // steps_per_iter`` (24 steps per
PPO iteration by default); ``common_step_counter`` is not randomized by ``train.py``.

Terms:
- :class:`assist_decay`: once per iteration, EMA of the assisted envs' success rate; decay x0.997 above 0.6
  (optional slow recovery). Replaces the AGILE per-reset height rule, which the harness itself can satisfy.
- :class:`reward_weight_ramp`: linear or log-space ramp of a reward weight between two iterations.
- :class:`effort_beta_schedule`: Stage-1 effort x1.2 (hips/knees/shoulders), milestone detection (assist = 0 and
  zero-assist success >= 70 %), then effort -> 1.0 and beta -> 0.8 over 1k iterations; per-env motor-strength DR.
  Uses ``set_effort_scale``. Sets ``getup_state.stage`` (0 = Stage 1, 1 = transition, 2 = done).
- :class:`domain_rand_widen`: when ``getup_state.stage >= 1`` (or ``force``), overwrite event-term params with the
  wide ranges and re-apply startup-mode terms once.
- :class:`category_reweighting`: every 200 iterations, ``update_category_probs`` with p_c ∝ (1 - success_c) + 0.1.
- :class:`curriculum_checkpoint`: iteration offset + curriculum_state.json save/restore for --resume.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

from .state import _CATEGORIES, ensure_state, joint_tables, per_joint_values

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _iteration(env: ManagerBasedRLEnv, steps_per_iter: int) -> int:
    """PPO iteration = ``getup_state.iteration_offset`` (resume, see :class:`curriculum_checkpoint`) +
    ``common_step_counter // steps_per_iter``."""
    offset = int(getattr(ensure_state(env), "iteration_offset", 0))
    return offset + int(env.common_step_counter) // max(1, int(steps_per_iter))


def _as_ids(env: ManagerBasedRLEnv, env_ids) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


# --- assist decay -----------------------------------------------------------------------------------------------------


class assist_decay(ManagerTermBase):
    """Decay the assist harness scale once per PPO iteration from the assisted envs' **success** rate.

    The AGILE rule uses "episode max height > h* - 0.1" and decays x0.9998 per reset call. In practice the harness
    itself satisfied that metric (metric EMA 0.975 at success 0) and, with a reset on nearly every env step, the scale
    halved in ~160 iterations. Instead:

    - metric: fraction of finished **assisted** episodes that succeeded (``is_standing`` held >= 1 s), accumulated on the
      GPU over one iteration (``steps_per_iter`` env steps, from ``common_step_counter``), taken from the
      **deterministic** (policy-mean) envs (sampled-action success is ~0 when the exploration std is large); from
      the sampled envs only if the run has no deterministic envs (logged as ``deterministic_source``); an iteration
      without finished deterministic assisted episodes holds the EMA and the scale;
    - once per iteration with >= 1 such episode: ``ema <- (1 - ema_alpha) ema + ema_alpha metric``;
    - decision per iteration: if ``ema > threshold``: ``scale *= decay`` (0.997: half-life ~230 iterations);
      below ``disable_below`` the scale snaps to 0;
    - optional recovery (``allow_recover``): if ``scale < recover_below_scale`` and the tracker's unassisted success EMA
      (``getup_state.curr_success_unassisted_ema``, deterministic) is below ``recover_unassisted_below``,
      ``scale = min(1, max(scale, disable_below) * recover_factor)`` instead
      (the policy relies on the harness: give some back slowly). Recovery takes precedence over decay and is disabled
      once ``getup_state.stage >= 1`` (the assist stays off from the milestone on).

    Logged every call: ``scale``, ``success_ema`` (assisted), ``unassisted_ema``, ``decision`` (-1 decay, 0 hold,
    +1 recover) and ``episodes`` (assisted episodes in the last evaluated iteration).
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._term = env.action_manager.get_term(cfg.params.get("action_name", "assist"))
        self.scale = float(self._term.scale)
        self.ema = 0.0
        # [sampled, deterministic] counts of finished assisted episodes and their successes (GPU, no sync per call)
        self._n = torch.zeros(2, device=env.device)
        self._s = torch.zeros(2, device=env.device)
        self._source = 0.0
        self._last_iter = 0
        self._decision = 0.0
        self._episodes = 0.0
        self._unassisted = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        action_name: str = "assist",
        ema_alpha: float = 0.05,
        threshold: float = 0.6,
        decay: float = 0.997,
        disable_below: float = 0.02,
        allow_recover: bool = True,
        recover_factor: float = 1.002,
        recover_below_scale: float = 0.3,
        recover_unassisted_below: float = 0.1,
        recover_min_episodes: float = 30.0,
        steps_per_iter: int = 24,
    ) -> dict[str, float]:
        st = ensure_state(env)
        ids = _as_ids(env, env_ids)
        mask = st.control_started[ids] & st.assist_enabled[ids]
        det = st.deterministic[ids]
        succ = st.success[ids] & mask
        self._n += torch.stack([(mask & ~det).sum(), (mask & det).sum()]).float()
        self._s += torch.stack([(succ & ~det).sum(), (succ & det).sum()]).float()
        it = _iteration(env, steps_per_iter)
        if it != self._last_iter:
            self._last_iter = it
            n_s, n_d = self._n.tolist()  # the one host sync of this term per iteration
            s_s, s_d = self._s.tolist()
            # Deterministic (policy-mean) episodes drive the decay. If deterministic envs exist but none of their
            # assisted episodes finished this iteration, hold (n = 0); sampled episodes are used only when the run has
            # no deterministic envs at all.
            use_det = n_d > 0 or bool(st.deterministic.any())
            n, s_sum = (n_d, s_d) if use_det else (n_s, s_s)
            self._source = 1.0 if use_det else 0.0
            self._episodes = n
            self._decision = 0.0
            self._unassisted = float(st.curr_success_unassisted_ema)
            if n > 0:
                self.ema = (1.0 - ema_alpha) * self.ema + ema_alpha * s_sum / n
                recover = (
                    allow_recover
                    and st.stage == 0  # the assist stays off from the milestone on
                    and self.scale < recover_below_scale
                    and self._unassisted < recover_unassisted_below
                    # not on a (near-)empty unassisted window, e.g. right after a resume
                    and float(st.curr_unassisted_n) >= recover_min_episodes
                )
                if recover:
                    self.scale = min(1.0, max(self.scale, disable_below) * recover_factor)
                    self._decision = 1.0
                elif self.ema > threshold and self.scale > 0.0:
                    self.scale *= decay
                    if self.scale < disable_below:
                        self.scale = 0.0
                    self._decision = -1.0
            self._n.zero_()
            self._s.zero_()
        self._term.scale = self.scale
        st.assist_scale = self.scale
        return {
            "scale": self.scale,
            "success_ema": self.ema,
            "unassisted_ema": self._unassisted,
            "decision": self._decision,
            "episodes": self._episodes,
            "deterministic_source": self._source,
        }

    def state_dict(self) -> dict[str, Any]:
        return {"scale": self.scale, "ema": self.ema}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.scale = float(d["scale"])
        self.ema = float(d["ema"])
        self._term.scale = self.scale
        ensure_state(self._env).assist_scale = self.scale


# --- reward weight ramps ----------------------------------------------------------------------------------------------


class reward_weight_ramp(ManagerTermBase):
    """Ramp a reward term's weight from its configured value to ``end_weight`` between two PPO iterations."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.start_weight = float(env.reward_manager.get_term_cfg(cfg.params["term_name"]).weight)
        if cfg.params.get("log_space", True):
            end = float(cfg.params["end_weight"])
            if self.start_weight == 0.0 or end == 0.0 or (self.start_weight > 0) != (end > 0):
                raise ValueError(
                    f"reward_weight_ramp({cfg.params['term_name']}): log ramp needs same-sign non-zero weights, got"
                    f" {self.start_weight} -> {end}"
                )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        term_name: str,
        start_iter: int,
        end_iter: int,
        end_weight: float,
        log_space: bool = True,
        steps_per_iter: int = 24,
    ) -> float:
        it = _iteration(env, steps_per_iter)
        u = min(1.0, max(0.0, (it - start_iter) / max(1, end_iter - start_iter)))
        if log_space:
            w = math.copysign(
                math.exp(math.log(abs(self.start_weight)) + u * (math.log(abs(end_weight)) - math.log(abs(self.start_weight)))),
                self.start_weight,
            )
        else:
            w = self.start_weight + u * (end_weight - self.start_weight)
        env.reward_manager.get_term_cfg(term_name).weight = w
        return w


# --- effort limits, action bound and motor strength -------------------------------------------------------------------

_STRONG_KEYS = (".*_hip_.*_joint", ".*_knee_joint", ".*_shoulder_.*_joint")
_OTHER_KEYS = (".*_(elbow|wrist_yaw|ankle_pitch|ankle_roll)_joint", "waist_yaw_joint")


class effort_beta_schedule(ManagerTermBase):
    """Effort-limit and action-bound curriculum.

    Stage 0: effort ``stage1_scale`` on hips/knees/shoulders, 1.0 elsewhere; beta = ``beta_start``.
    Milestone (-> stage 1): the assist scale has been 0 for at least ``min_iters_after_assist`` consecutive PPO
    iterations and ``getup_state.curr_success_ema`` (deterministic envs) >= ``milestone_success`` (checked once per
    iteration). If the assist comes back (> 0) the count restarts.
    Stage 1: over ``transition_iters`` iterations, effort -> ``final_scale`` (all scaled groups) and
    beta -> ``beta_end``. Stage 2: final values.
    Action bound: ``set_bound_scale`` with the uniform stage values (strong / other), never the motor strength.
    Motor strength: in stage >= 1 each resetting env draws a factor from ``strength_range_wide`` quantized to
    ``num_strength_buckets`` levels (optionally ``strength_focus_frac`` of the draws from ``strength_focus_range``),
    multiplied into every joint's effort **clip** only (per reset call; this path syncs).
    In stage 0 (``strength_range`` degenerate) the effort scale is identical for all envs, so ``set_effort_scale`` is
    called once for all envs, only when the value changes (effort scales persist across resets; no GPU->host sync).
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        joint_tables(env)  # capture nominal limits before any scaling
        st = ensure_state(env)
        st.stage = int(cfg.params.get("start_stage", 0))
        self._stage_start_iter = 0
        self._iter_at_assist_zero: int | None = None
        self._last_iter = -1
        self._applied: tuple | None = None
        self._action = env.action_manager.get_term(cfg.params.get("action_name", "joint_pos"))
        self._set_effort_scale = _resolve_set_effort_scale()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        action_name: str = "joint_pos",
        stage1_scale: float = 1.2,
        final_scale: float = 1.0,
        beta_start: float = 1.0,
        beta_end: float = 0.8,
        transition_iters: int = 1000,
        milestone_success: float = 0.7,
        min_iters_after_assist: int = 50,
        strength_range: tuple[float, float] = (1.0, 1.0),
        strength_range_wide: tuple[float, float] = (0.85, 1.05),
        num_strength_buckets: int = 5,
        strength_focus_range: tuple[float, float] | None = None,
        strength_focus_frac: float = 0.0,
        start_stage: int = 0,
        steps_per_iter: int = 24,
    ) -> dict[str, float]:
        st = ensure_state(env)
        it = _iteration(env, steps_per_iter)
        # milestone detection, once per iteration (the only host sync of this term in stage 0)
        if st.stage == 0 and it != self._last_iter:
            if st.assist_scale > 0.0:
                self._iter_at_assist_zero = None
            else:
                if self._iter_at_assist_zero is None:
                    self._iter_at_assist_zero = it
                if it - self._iter_at_assist_zero >= min_iters_after_assist and float(st.curr_success_ema) >= milestone_success:
                    st.stage = 1
                    self._stage_start_iter = it
        self._last_iter = it
        if st.stage == 1 and it - self._stage_start_iter >= transition_iters:
            st.stage = 2
        # schedule values
        if st.stage == 0:
            u = 0.0
        elif st.stage == 1:
            u = min(1.0, (it - self._stage_start_iter) / max(1, transition_iters))
        else:
            u = 1.0
        strong = stage1_scale + u * (final_scale - stage1_scale)
        other = 1.0 + u * (final_scale - 1.0)
        beta = beta_start + u * (beta_end - beta_start)
        self._action.beta = beta
        # the action bound follows only the uniform stage scale; motor strength (below) scales the clip only, so the
        # motor-strength randomization teaches robustness to weak motors
        bound_key = (strong, other)
        if bound_key != getattr(self, "_bound_applied", None):
            bound = {k: strong for k in _STRONG_KEYS}
            bound.update({k: other for k in _OTHER_KEYS})
            self._action.set_bound_scale(bound)
            self._bound_applied = bound_key
        lo, hi = strength_range_wide if st.stage >= 1 else strength_range
        if hi <= lo:
            # uniform effort scale for all envs: apply once per change, to all envs (no per-reset work, no sync)
            key = (strong, other, lo)
            if key != self._applied:
                scale = {k: strong * lo for k in _STRONG_KEYS}
                scale.update({k: other * lo for k in _OTHER_KEYS})
                self._set_effort_scale(env, scale, env_ids=None)
                self._applied = key
        else:
            # per-env motor strength buckets for the resetting envs
            self._applied = None
            ids = _as_ids(env, env_ids)
            nb = max(1, int(num_strength_buckets))
            levels = torch.linspace(lo, hi, nb).tolist() if nb > 1 else [0.5 * (lo + hi)]
            probs = [1.0 / nb] * nb
            if strength_focus_range is not None and strength_focus_frac > 0.0:
                # oversample a sub-range (e.g. weak motors): `strength_focus_frac` of the draws come from it
                f_lo, f_hi = strength_focus_range
                f_levels = torch.linspace(f_lo, f_hi, nb).tolist() if nb > 1 else [0.5 * (f_lo + f_hi)]
                probs = [(1.0 - strength_focus_frac) / nb] * nb + [strength_focus_frac / nb] * nb
                levels = levels + f_levels
            p_t = torch.tensor(probs, device=env.device)
            bucket = torch.multinomial(p_t, ids.numel(), replacement=True)
            for b, strength in enumerate(levels):
                sel = ids[bucket == b]
                if sel.numel() == 0:
                    continue
                scale = {k: strong * strength for k in _STRONG_KEYS}
                scale.update({k: other * strength for k in _OTHER_KEYS})
                self._set_effort_scale(env, scale, env_ids=sel)
        return {"stage": float(st.stage), "effort_strong": strong, "effort_other": other, "beta": beta}

    def state_dict(self) -> dict[str, Any]:
        st = ensure_state(self._env)
        return {
            "stage": int(st.stage),
            "stage_start_iter": int(self._stage_start_iter),
            "iter_at_assist_zero": self._iter_at_assist_zero,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        ensure_state(self._env).stage = int(d["stage"])
        self._stage_start_iter = int(d["stage_start_iter"])
        self._iter_at_assist_zero = None if d.get("iter_at_assist_zero") is None else int(d["iter_at_assist_zero"])
        self._applied = None
        self._bound_applied = None


def _resolve_set_effort_scale():
    """``getup_actuators.set_effort_scale``, or a fallback that writes ``actuator.effort_limit`` directly."""
    try:
        from isaac_asimov.assets.robots.getup_actuators import set_effort_scale as w1_set
    except ImportError:
        w1_set = None

    def _shim(env, scale: dict[str, float], env_ids: torch.Tensor):
        import re

        tab = joint_tables(env)
        asset = env.scene["robot"]
        for act in asset.actuators.values():
            for j, name in enumerate(act.joint_names):
                hits = [v for k, v in scale.items() if re.fullmatch(k, name)]
                if hits:
                    gid = tab.names.index(name)
                    rows = slice(None) if env_ids is None else env_ids
                    act.effort_limit[rows, j] = tab.tau_max[gid] * hits[0]

    def _call(env, scale, env_ids):
        if w1_set is not None:
            try:
                return w1_set(env, scale, env_ids=env_ids)
            except NotImplementedError:
                pass
        return _shim(env, scale, env_ids)

    return _call


# --- fixed effort scale for play / evaluation (no curricula) --------------------------------------------------------

EFFORT_PRESETS: dict[str, dict[str, float]] = {
    "nominal": {k: 1.0 for k in _STRONG_KEYS + _OTHER_KEYS},
    "stage1": {**{k: 1.2 for k in _STRONG_KEYS}, **{k: 1.0 for k in _OTHER_KEYS}},
    "stageB": {k: 0.9 for k in _STRONG_KEYS + _OTHER_KEYS},
}
"""Named effort settings: ``stage1`` = the Stage-1 training setting (hips/knees/shoulders x1.2)."""


def resolve_effort_scale(scale: float | str | dict[str, float]) -> float | dict[str, float]:
    """``float`` (all joints), preset name (:data:`EFFORT_PRESETS`) or ``{joint regex: scale}`` for
    ``set_effort_scale`` (joints not matched keep 1.0 at startup)."""
    if isinstance(scale, str):
        if scale not in EFFORT_PRESETS:
            raise ValueError(f"unknown effort preset '{scale}', expected a float, a dict or one of {list(EFFORT_PRESETS)}")
        return dict(EFFORT_PRESETS[scale])
    if isinstance(scale, dict):
        return {str(k): float(v) for k, v in scale.items()}
    return float(scale)


def apply_play_effort_scale(
    env: ManagerBasedRLEnv,
    env_ids,
    scale: float | str | dict | None = None,
    motor_strength: float | dict | None = None,
) -> None:
    """Event term (mode ``startup``) for envs without curricula (PLAY / eval).

    ``scale`` (default ``env.cfg.play_effort_scale``) is a **trained condition**: it sets both the action bound
    (``set_bound_scale``) and the actuator effort clip. ``motor_strength`` (default ``env.cfg.play_motor_strength``) is a
    stress test: it multiplies the clip only (e.g. the 0.9x weak-motor check with the bound at the certified value).
    Also callable directly after ``gym.make`` (then pass both explicitly or set the cfg fields first).
    """
    if scale is None:
        scale = getattr(env.cfg, "play_effort_scale", None)
    if motor_strength is None:
        motor_strength = getattr(env.cfg, "play_motor_strength", None)
    if scale is None and motor_strength is None:
        return
    action = env.action_manager.get_term("joint_pos")
    names = list(action._joint_names)
    bound = per_joint_values(resolve_effort_scale(scale if scale is not None else 1.0), names)
    strength = per_joint_values(resolve_effort_scale(motor_strength if motor_strength is not None else 1.0), names)
    action.set_bound_scale(torch.tensor(bound, device=env.device))
    clip = {n: b * m for n, b, m in zip(names, bound, strength)}
    _resolve_set_effort_scale()(env, clip, None)


def action_contract(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> dict:
    """The live action contract of ``env``: per-joint s_j, beta, bound scale, LPF settings, clip, joint order.
    Evaluation and ONNX export read this and assert the exported metadata against it."""
    return env.action_manager.get_term(action_name).contract()


# --- domain randomization widening ------------------------------------------------------------------------------------


class domain_rand_widen(ManagerTermBase):
    """Switch event terms to their wide DR ranges once ``getup_state.stage >= 1`` (or immediately with ``force``).

    ``overrides``: ``{event_term_name: {param_name: value}}``. Startup-mode terms are re-applied once to all envs;
    reset/interval terms pick the new params up on their next call. Terms missing from the event manager are skipped.
    ``randomize_rigid_body_material`` samples its material buckets only in ``__init__``, so its buckets are re-sampled
    from the new ranges on the term instance before it is re-applied.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.applied = False

    def __call__(
        self, env: ManagerBasedRLEnv, env_ids, overrides: dict[str, dict[str, Any]], force: bool = False
    ) -> float:
        st = ensure_state(env)
        if not self.applied and (force or st.stage >= 1):
            em = env.event_manager
            names = set()
            for mode in em.available_modes:
                names.update(em.active_terms[mode])
            for term_name, params in overrides.items():
                if term_name not in names:
                    continue
                tcfg = em.get_term_cfg(term_name)
                tcfg.params.update(params)
                _resample_material_buckets(tcfg)
                if tcfg.mode == "startup":
                    tcfg.func(env, None, **tcfg.params)
            self.applied = True
        return float(self.applied)


def _resample_material_buckets(tcfg) -> None:
    """Re-sample ``material_buckets`` of a ``randomize_rigid_body_material`` instance from ``tcfg.params`` (same rule as
    its ``__init__``, Isaac Lab ``envs/mdp/events.py:225-238``). No-op for other terms."""
    term = tcfg.func
    if not hasattr(term, "material_buckets"):
        return
    import isaaclab.utils.math as math_utils

    p = tcfg.params
    ranges = torch.tensor(
        [p.get("static_friction_range", (1.0, 1.0)), p.get("dynamic_friction_range", (1.0, 1.0)),
         p.get("restitution_range", (0.0, 0.0))],
        device="cpu",
    )  # fmt: skip
    buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (int(p.get("num_buckets", 1)), 3), device="cpu")
    if p.get("make_consistent", False):
        buckets[:, 1] = torch.min(buckets[:, 0], buckets[:, 1])
    term.material_buckets = buckets


# --- category reweighting ---------------------------------------------------------------------------------------------


class category_reweighting(ManagerTermBase):
    """Every ``every_iters`` iterations: p_c ∝ (1 - success_c) + offset, floor, renormalize
    (``update_category_probs``)."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._last_iter = 0
        self._probs: dict[str, float] = {}
        try:
            self._probs = dict(env.event_manager.get_term_cfg(cfg.params.get("event_term", "reset_fallen")).params.get(
                "category_probs", {}))
        except (ValueError, KeyError, AttributeError):
            self._probs = {}

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        every_iters: int = 200,
        floor: float = 0.03,
        offset: float = 0.1,
        min_episodes: int = 500,
        min_category_episodes: float = 20.0,
        event_term: str = "reset_fallen",
        steps_per_iter: int = 24,
    ) -> dict[str, float]:
        st = ensure_state(env)
        it = _iteration(env, steps_per_iter)
        if (
            it - self._last_iter >= every_iters
            and int(st.episodes_finished) >= min_episodes
            and self._windows_filled(env, event_term, min_category_episodes)
        ):
            self._last_iter = it
            succ = st.curr_success_by_category.detach().clone()  # deterministic envs
            probs = None
            try:
                from .resets import update_category_probs

                probs = update_category_probs(env, succ, floor=floor, offset=offset)
            except (ImportError, NotImplementedError):
                probs = None
            if probs is None:  # fallback if update_category_probs is unavailable
                w = (1.0 - succ.clamp(0.0, 1.0)) + offset
                p = w / w.sum()
                p = p.clamp(min=floor)
                p = p / p.sum()
                probs = {c: float(p[i]) for i, c in enumerate(_CATEGORIES)}
                try:
                    env.event_manager.get_term_cfg(event_term).params["category_probs"] = dict(probs)
                except (ValueError, KeyError):
                    pass
            self._probs = dict(probs)
        return {f"p_{k}": float(v) for k, v in self._probs.items()}

    @staticmethod
    def _windows_filled(env: ManagerBasedRLEnv, event_term: str, min_n: float) -> bool:
        """Every category with p > 0 has >= ``min_n`` episodes in the window the reweighting reads."""
        try:
            probs = env.event_manager.get_term_cfg(event_term).params["category_probs"]
        except (ValueError, KeyError):
            return True
        n = ensure_state(env).curr_category_n.tolist()
        return all(n[i] >= min_n for i, c in enumerate(_CATEGORIES) if float(probs.get(c, 0.0)) > 0.0)

    def state_dict(self) -> dict[str, Any]:
        return {"last_iter": int(self._last_iter), "category_probs": dict(self._probs)}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self._last_iter = int(d["last_iter"])
        probs = {k: float(v) for k, v in d["category_probs"].items()}
        if probs:
            self._probs = dict(probs)
            try:
                current = self._env.event_manager.get_term_cfg(self.cfg.params.get("event_term", "reset_fallen")).params
                current["category_probs"].clear()
                current["category_probs"].update(probs)
            except (ValueError, KeyError):
                pass


# --- curriculum persistence -------------------------------------------------------------------------------------------


class curriculum_checkpoint(ManagerTermBase):
    """Persist and restore curriculum state across ``--resume``. Declare it **last** in the curriculum cfg.

    - Iteration offset: ``getup_state.iteration_offset`` = ``start_iteration`` if >= 0, else the ``iteration`` stored in
      ``restore_path``, else 0. Every schedule uses ``offset + common_step_counter // steps_per_iter``.
    - Save: whenever the iteration is a multiple of ``save_interval`` (the runner's checkpoint cadence), writes
      ``<log_dir>/curriculum_state_<it>.json`` and ``<log_dir>/curriculum_state.json`` (latest) with the ``state_dict()``
      of every curriculum term that has one (assist scale/EMA, effort stage and its start iteration, category
      probabilities) plus ``iteration``, ``steps_per_iter`` and the live ``action_contract`` (s_j, beta, bound
      scale, LPF, clip, joint order).
    - Restore: if ``restore_path`` is set, the saved states are loaded into the terms on the first call.
    - Startup check: on the first training step, ``steps_per_iter`` must equal ``num_steps_per_env`` in
      ``<log_dir>/params/agent.yaml`` (written by ``train.py``); otherwise every schedule would drift, so it raises.

    Not persisted (restart from scratch on resume): the tracker's success windows (``GetUp/success_*``, ``success_ema``)
    and ``getup_state.episodes_finished``.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        import json
        import os

        super().__init__(cfg, env)
        st = ensure_state(env)
        path = os.path.expanduser(str(cfg.params.get("restore_path", "") or ""))
        self._data: dict[str, Any] | None = None
        if path:
            with open(path) as f:
                self._data = json.load(f)
        start = int(cfg.params.get("start_iteration", -1))
        if start < 0:
            start = int(self._data["iteration"]) if self._data else 0
        st.iteration_offset = start
        self._restored = False
        self._checked = False
        self._last_saved: int | None = None

    def _stateful_terms(self, env: ManagerBasedRLEnv) -> dict[str, Any]:
        cm = env.curriculum_manager
        out = {}
        for name, tcfg in zip(cm._term_names, cm._term_cfgs):
            term = tcfg.func
            if term is not self and hasattr(term, "state_dict") and hasattr(term, "load_state_dict"):
                out[name] = term
        return out

    def _check_steps_per_iter(self, env: ManagerBasedRLEnv, steps_per_iter: int) -> None:
        import os
        import re

        log_dir = getattr(env.cfg, "log_dir", None)
        if not log_dir:
            return
        path = os.path.join(log_dir, "params", "agent.yaml")
        if not os.path.exists(path):
            return
        m = re.search(r"^num_steps_per_env:\s*(\d+)", open(path).read(), re.M)
        if m and int(m.group(1)) != int(steps_per_iter):
            raise ValueError(
                f"[getup] runner num_steps_per_env={m.group(1)} but the curricula use steps_per_iter={steps_per_iter}"
                " (getup_env_cfg.STEPS_PER_ITER); every iteration-based schedule would drift. Make them equal."
            )
        for name, tcfg in zip(env.curriculum_manager._term_names, env.curriculum_manager._term_cfgs):
            v = tcfg.params.get("steps_per_iter")
            if v is not None and int(v) != int(steps_per_iter):
                raise ValueError(f"[getup] curriculum term '{name}' has steps_per_iter={v} != {steps_per_iter}")

    def _save(self, env: ManagerBasedRLEnv, it: int, steps_per_iter: int) -> None:
        import json
        import os

        log_dir = getattr(env.cfg, "log_dir", None)
        if not log_dir:
            return
        data = {
            "iteration": it,
            "steps_per_iter": steps_per_iter,
            "terms": {name: term.state_dict() for name, term in self._stateful_terms(env).items()},
            # the live action contract at this iteration (export / eval assert against it)
            "action_contract": action_contract(env),
        }
        os.makedirs(log_dir, exist_ok=True)
        for fname in (f"curriculum_state_{it}.json", "curriculum_state.json"):
            tmp = os.path.join(log_dir, f".{fname}.tmp")
            with open(tmp, "w") as f:
                json.dump(data, f, indent=1)
            os.replace(tmp, os.path.join(log_dir, fname))

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        save_interval: int = 250,
        restore_path: str = "",
        start_iteration: int = -1,
        steps_per_iter: int = 24,
    ) -> dict[str, float]:
        if not self._restored:
            self._restored = True
            if self._data:
                terms = self._stateful_terms(env)
                for name, d in self._data.get("terms", {}).items():
                    if name in terms:
                        terms[name].load_state_dict(d)
        if not self._checked and env.common_step_counter > 0:
            self._checked = True
            self._check_steps_per_iter(env, steps_per_iter)
        it = _iteration(env, steps_per_iter)
        if env.common_step_counter > 0 and it % max(1, int(save_interval)) == 0 and it != self._last_saved:
            self._last_saved = it
            self._save(env, it, steps_per_iter)
        return {"iteration": float(it), "restored": float(self._data is not None)}
