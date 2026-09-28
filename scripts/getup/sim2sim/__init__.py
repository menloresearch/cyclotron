# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""MuJoCo sim2sim harness for Asimov-1 get-up / walking ONNX policies.

Deliberately free of any ``isaaclab`` / ``isaac_asimov`` (torch) import so the whole package runs in a bare
``mujoco`` + ``onnxruntime`` venv, independent of the Isaac Lab environment. Constants that mirror Isaac Lab's own
frozen interfaces (joint order, actuator gains, slot layout, action contract) are duplicated as plain Python
literals in :mod:`constants` rather than imported, and cross-checked against the live Isaac source by
``scripts/getup/sim2sim/capture_isaac_obs.py`` (which *does* need Isaac Lab and is run separately, see its
module docstring).
"""
