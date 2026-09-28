# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""The get-up one-pole low-pass action filter: ``y <- y + alpha * (x - y)``.

Alpha depends on the rate it's applied at (``dt / (RC + dt)``, RC = 1/(2*pi*10 Hz)): 0.24 at a 200 Hz
rate, 0.557 at the 50 Hz policy rate the get-up action term actually filters at -- see
:data:`constants.GETUP_LPF_ALPHA`. This class takes whatever alpha it's given;
callers own picking the right one.
"""

from __future__ import annotations

import numpy as np


class OnePoleLPF:
    """Vector one-pole IIR filter, applied once per call (call it once per policy tick, not per physics step,
    unless ``lpf_per_substep`` -- see :mod:`actions`)."""

    def __init__(self, dim: int, alpha: float = 0.24):
        self.alpha = float(alpha)
        self.y = np.zeros(dim, dtype=float)

    def reset(self, value: np.ndarray | float = 0.0) -> None:
        self.y[:] = value

    def step(self, x: np.ndarray) -> np.ndarray:
        self.y = self.y + self.alpha * (np.asarray(x, dtype=float) - self.y)
        return self.y


class TimeDelayBuffer:
    """Per-signal fixed-lag delay buffer, one independent lag (in *steps of this buffer's own call rate*) per
    element of the last axis.

    Mirrors Isaac Lab's ``DelayBuffer``/``DelayedPDActuator`` semantics: a circular buffer of the last
    ``max_lag + 1`` pushed values, from which the value ``lag`` calls ago is read back. ``lag`` is drawn once (per
    element) at construction / :meth:`resample` and held fixed until the next resample (an episode reset).

    On reset, the buffer does *not* zero-pad: both Isaac mechanisms this mirrors avoid a startup transient from
    stale zeros -- ``delayed_obs`` explicitly repeats the first post-reset value into every slot on its first call,
    and ``DelayBuffer``/``CircularBuffer`` clamp a not-yet-available lag to "the latest pushed value" (its own
    docstring: "if the requested delay is larger than the number of buffered data points ... returns the latest
    data"). :meth:`reset` with no ``value`` defers this to the next :meth:`push_and_read` (lazy fill, matching
    ``delayed_obs``); :meth:`reset` with a ``value`` fills immediately (used by the actuator, which knows ``q0`` at
    reset time).
    """

    def __init__(self, dim: int, min_lag_steps: np.ndarray | int, max_lag_steps: np.ndarray | int, rng: np.random.Generator):
        self.dim = dim
        self._rng = rng
        self.min_lag = np.broadcast_to(np.asarray(min_lag_steps, dtype=int), (dim,)).copy()
        self.max_lag = np.broadcast_to(np.asarray(max_lag_steps, dtype=int), (dim,)).copy()
        self._cap = int(self.max_lag.max()) + 1
        self._buf = np.zeros((self._cap, dim), dtype=float)
        self.lag = np.zeros(dim, dtype=int)
        self._primed = False
        self.resample()

    def resample(self) -> None:
        self.lag = np.array(
            [self._rng.integers(lo, hi + 1) if hi >= lo else 0 for lo, hi in zip(self.min_lag, self.max_lag)],
            dtype=int,
        )

    def reset(self, value: np.ndarray | float | None = None) -> None:
        if value is None:
            self._primed = False
        else:
            self._buf[:] = value
            self._primed = True

    def push_and_read(self, value: np.ndarray) -> np.ndarray:
        if not self._primed:
            self._buf[:] = value
            self._primed = True
        else:
            self._buf = np.roll(self._buf, shift=1, axis=0)
            self._buf[0] = value
        return self._buf[self.lag, np.arange(self.dim)]


class SharedLagDelayBuffer(TimeDelayBuffer):
    """Like :class:`TimeDelayBuffer`, but every element of the last axis shares ONE random lag (matching
    ``tasks/locomotion/mdp/observations.py::delayed_obs``, which draws one lag per env, not per feature)."""

    def resample(self) -> None:
        lo, hi = int(self.min_lag[0]), int(self.max_lag[0])
        lag = int(self._rng.integers(lo, hi + 1)) if hi >= lo else 0
        self.lag = np.full(self.dim, lag, dtype=int)


class TermHistory:
    """Per-observation-term history buffer, matching ``isaaclab.utils.buffers.CircularBuffer`` semantics used by
    ``ObservationManager`` for a group's ``history_length``: oldest entry first, most recent last, flattened in that
    order; a :meth:`reset` zeros the buffer (not "repeat the current value") so a just-reset episode's history is
    zero-padded until enough new ticks have been appended -- exactly Isaac's behaviour after ``env.reset()``.
    """

    def __init__(self, dim: int, length: int):
        self.dim = dim
        self.length = length
        self._buf = np.zeros((length, dim), dtype=float)

    def reset(self) -> None:
        self._buf[:] = 0.0

    def append(self, value: np.ndarray) -> np.ndarray:
        self._buf = np.roll(self._buf, shift=-1, axis=0)
        self._buf[-1] = value
        return self.flatten()

    def flatten(self) -> np.ndarray:
        return self._buf.reshape(-1)
