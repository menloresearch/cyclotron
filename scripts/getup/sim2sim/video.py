# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Offscreen MP4 rendering via ``mujoco.Renderer`` (set ``MUJOCO_GL=egl`` on a headless machine;
``osmesa``/``glfw`` also work locally with a display).

:class:`TrackingCamera` matches the Isaac eval camera (``scripts/getup/_common.py::camera_pose_for_height``, a 3/4
side-angle view): a MuJoCo *free* camera (``mjvCamera``, manual
azimuth/elevation/distance -- not one of the MJCF's fixed ``<camera>`` defs) whose ``lookat`` tracks
``pelvis_link``'s projected (x, y) position every frame (held at a fixed height, not the body's own z, so the
lookat doesn't dive to the floor when the robot falls), azimuth yaw-relative to the torso heading, both
EMA-smoothed so a fast fall or a noisy per-tick heading doesn't jitter the shot.
"""

from __future__ import annotations

import numpy as np

import mujoco


class TrackingCamera:
    """A 3/4 front-side tracking shot: eye at ``horizontal_distance`` from the pelvis (projected to the ground),
    ``eye_height`` up, looking at ``(pelvis_x, pelvis_y, lookat_z)``, azimuth offset from -- and EMA-following --
    the torso's own heading (so the camera stays at the same relative angle as the robot turns)."""

    def __init__(
        self,
        azimuth_offset_deg: float = 35.0,
        horizontal_distance: float = 2.4,
        eye_height: float = 0.9,
        lookat_z: float = 0.35,
        ema_alpha: float = 0.06,
    ):
        self.azimuth_offset_deg = azimuth_offset_deg
        self.horizontal_distance = horizontal_distance
        self.eye_height = eye_height
        self.lookat_z = lookat_z
        self.ema_alpha = ema_alpha
        self._ema_pos_xy: np.ndarray | None = None
        self._ema_heading: np.ndarray | None = None  # unit 2-vector, EMA'd in vector space (no angle wraparound)

        self.mjv_camera = mujoco.MjvCamera()
        self.mjv_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        height_diff = eye_height - lookat_z
        self.mjv_camera.distance = float(np.hypot(horizontal_distance, height_diff))
        # MuJoCo's mjvCamera.elevation convention: NEGATIVE tilts the view down. The eye is above the lookat
        # point (height_diff > 0 for a normal "look down at the robot" shot), so elevation must be negative --
        # confirmed empirically (a positive sign here pointed the camera up into the empty sky/background and
        # left the ground/robot cut off in the bottom half of every frame).
        self.mjv_camera.elevation = -float(np.degrees(np.arctan2(height_diff, horizontal_distance)))

    def reset(self) -> None:
        self._ema_pos_xy = None
        self._ema_heading = None

    def update(self, pelvis_xy: np.ndarray, torso_heading_rad: float) -> None:
        heading_vec = np.array([np.cos(torso_heading_rad), np.sin(torso_heading_rad)])
        if self._ema_pos_xy is None:
            self._ema_pos_xy = np.asarray(pelvis_xy, dtype=float).copy()
            self._ema_heading = heading_vec
        else:
            a = self.ema_alpha
            self._ema_pos_xy = a * np.asarray(pelvis_xy, dtype=float) + (1 - a) * self._ema_pos_xy
            self._ema_heading = a * heading_vec + (1 - a) * self._ema_heading
            n = np.linalg.norm(self._ema_heading)
            if n > 1e-9:
                self._ema_heading = self._ema_heading / n

        yaw_deg = float(np.degrees(np.arctan2(self._ema_heading[1], self._ema_heading[0])))
        self.mjv_camera.azimuth = yaw_deg + self.azimuth_offset_deg
        self.mjv_camera.lookat[:] = [self._ema_pos_xy[0], self._ema_pos_xy[1], self.lookat_z]


def torso_heading_rad(quat_wxyz: np.ndarray) -> float:
    """Yaw of the pelvis's local +x (forward) axis, projected to the world XY plane, from a wxyz quaternion."""
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class EpisodeRecorder:
    def __init__(
        self,
        model: "mujoco.MjModel",
        width: int = 1280,
        height: int = 720,
        camera: "str | int | TrackingCamera | None" = None,
    ):
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.tracking_camera = camera if isinstance(camera, TrackingCamera) else None
        self.camera = camera.mjv_camera if self.tracking_camera is not None else (camera if camera is not None else -1)
        self.frames: list[np.ndarray] = []

    def capture(self, data: "mujoco.MjData") -> None:
        self.renderer.update_scene(data, camera=self.camera)
        self.frames.append(self.renderer.render().copy())

    def save(self, path: str, fps: float) -> None:
        """Requires the ``imageio-ffmpeg`` package (for machines with no system ``ffmpeg`` on ``$PATH`` -- see also
        ``scripts/getup/_common.py::resolve_ffmpeg_exe``). Once it's installed, plain
        ``imageio.mimwrite`` resolves the ``.mp4`` extension to it automatically; do NOT pass ``plugin="ffmpeg"``
        here -- this imageio version's legacy-plugin wrapper forwards unrecognized ``mimwrite`` kwargs straight
        into ``FfmpegFormat.Writer._open()``, which raises ``TypeError: ... unexpected keyword argument 'plugin'``."""
        import imageio

        try:
            import imageio_ffmpeg  # noqa: F401  (registers the ffmpeg backend for the .mp4 extension)
        except ImportError as e:
            raise RuntimeError(
                "video export needs 'pip install imageio-ffmpeg' (no system ffmpeg on this server's $PATH)"
            ) from e
        imageio.mimwrite(path, self.frames, fps=fps, quality=8)

    def close(self) -> None:
        self.renderer.close()
