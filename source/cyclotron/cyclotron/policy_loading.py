"""Load only the policy from a checkpoint, for ``--export`` and ``--play``, after checking it fits the network the
current code builds.

Plain Python with no Isaac Lab imports, so it can run without Isaac Sim and in tests.
"""

from __future__ import annotations


def inference_load_cfg(runner_class: str) -> dict:
    """What ``runner.load`` must restore to run the policy: the actor, or the distilled student. The critic,
    optimizer and AMP discriminator are training-only, and may no longer fit after the code changed."""
    return {"student": True} if runner_class == "DistillationRunner" else {"actor": True}


def _listing(names: list[str], limit: int = 8) -> str:
    shown = ", ".join(names[:limit])
    return shown + (f" and {len(names) - limit} more" if len(names) > limit else "")


def _shape(tensor) -> list[int]:
    return list(getattr(tensor, "shape", ()))


def policy_shape_errors(saved: dict, current: dict) -> list[str]:
    """Compare a checkpoint's policy weights with the network the current code builds; describe each mismatch."""
    lines = []
    missing = [name for name in current if name not in saved]
    unexpected = [name for name in saved if name not in current]
    if missing or unexpected:
        lines.append("the network's layers changed:")
        if unexpected:
            lines.append(f"  only in the checkpoint: {_listing(unexpected)}")
        if missing:
            lines.append(f"  only in the current code: {_listing(missing)}")
    # With the same layers, the first and last weight matrices hold the policy's input and output sizes.
    matrices = [] if lines else [name for name in saved if len(_shape(saved[name])) == 2]
    for name in saved:
        if name not in current or _shape(saved[name]) == _shape(current[name]):
            continue
        old, new = _shape(saved[name]), _shape(current[name])
        line = f"{name}: checkpoint {old}, current code {new}"
        if matrices and name == matrices[0] and old[1:] != new[1:]:
            line += f" (the policy takes {old[1]} inputs; the current observations give {new[1]})"
        elif matrices and name == matrices[-1] and old[:1] != new[:1]:
            line += f" (the policy gives {old[0]} actions; the current code expects {new[0]})"
        lines.append(line)
    return lines


def load_policy(runner, checkpoint: str, runner_class: str, hint: str) -> None:
    """Restore only the policy from ``checkpoint`` into ``runner``, after checking it fits the current network.

    Raises ``ValueError`` with a plain description when it doesn't, instead of PyTorch's size-mismatch error, ending
    with ``hint`` (where to find the code the run was trained with, ``code_state.check_out_hint``).
    """
    import torch

    key = "student_state_dict" if runner_class == "DistillationRunner" else "actor_state_dict"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False).get(key)
    if saved is None:
        # Not the layout this runner saves (e.g. distilling from an RL checkpoint): let rsl_rl decide what to load.
        runner.load(checkpoint)
        return
    errors = policy_shape_errors(saved, runner.alg.get_policy().state_dict())
    if errors:
        raise ValueError(
            "The checkpoint's policy doesn't fit the network the current code builds:\n"
            + "\n".join(f"  {line}" for line in errors)
            + "\nThe policy's inputs or network changed since training; see the warning above.\n"
            + hint
        )
    runner.load(checkpoint, load_cfg=inference_load_cfg(runner_class))
