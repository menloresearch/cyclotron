# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Verify an exported get-up ONNX policy against saved torch reference outputs.

Deliberately has **no ``isaaclab`` or ``torch`` import** — only ``onnxruntime`` and ``numpy`` — so
it runs in a plain venv separate from the Isaac Sim training venv (keeping ``onnxruntime`` out of
the training environment). A small venv such as ``~/venvs/sim2sim`` with ``numpy`` and
``onnxruntime`` is all this script needs.

``scripts/getup/export_onnx.py`` (run in the Isaac venv) produces the two inputs this script
compares: the ``.onnx`` file itself, and a ``verify_samples.npz`` containing random observations
plus the exact in-memory torch model's outputs on them at export time. This script is the numeric
proof that the ONNX graph reproduces those outputs — i.e. that exporting didn't silently change the
policy's behavior — using the deployment runtime (``onnxruntime``, CPU execution
provider) rather than the pure-Python ``onnx.reference.ReferenceEvaluator``.

Usage:
    source ~/venvs/sim2sim/bin/activate
    python3 scripts/getup/verify_onnx_parity.py \\
        --onnx <run>/exported/policy.onnx --samples <run>/exported/verify_samples.npz
"""

from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify an exported ONNX policy against saved torch reference outputs.")
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--samples", type=str, required=True, help="verify_samples.npz from export_onnx.py.")
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    data = np.load(args.samples)
    obs, torch_actions = data["obs"], data["torch_actions"]

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    print(f"[getup-export] onnxruntime {ort.__version__}, providers={sess.get_providers()}")
    print(f"[getup-export] ONNX input {input_name!r} {input_shape}, output {output_name!r} {sess.get_outputs()[0].shape}")

    # This codebase exports every policy with a FIXED batch dimension of 1 (matching real firmware
    # inference: one observation at a time, no batching); a batched call raises `InvalidArgument:
    # Got invalid dimensions for input: obs ... Got: 32 Expected: 1`. Run one sample per call rather
    # than assuming a dynamic batch axis.
    batched = not (isinstance(input_shape[0], int) and input_shape[0] == 1)
    if batched:
        onnx_actions = sess.run([output_name], {input_name: obs})[0]
    else:
        onnx_actions = np.concatenate(
            [sess.run([output_name], {input_name: obs[i : i + 1]})[0] for i in range(obs.shape[0])], axis=0
        )

    max_abs_diff = float(np.max(np.abs(onnx_actions - torch_actions)))
    print(f"[getup-export] {obs.shape[0]} samples: max |onnxruntime - torch| = {max_abs_diff:.3e} (atol={args.atol:.3e})")

    assert onnx_actions.shape == torch_actions.shape, (
        f"shape mismatch: onnxruntime {onnx_actions.shape} vs. saved torch {torch_actions.shape}"
    )
    assert max_abs_diff <= args.atol, (
        f"onnxruntime output diverges from the saved torch reference by {max_abs_diff:.3e} > atol={args.atol:.3e} "
        "-- the exported ONNX graph does not reproduce the in-memory policy's behavior."
    )
    print("[getup-export] PASS: onnxruntime output matches the saved torch reference.")


if __name__ == "__main__":
    main()
