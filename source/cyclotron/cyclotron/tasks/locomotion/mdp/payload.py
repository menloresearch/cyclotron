# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Mass properties for a payload rigidly attached to a hand link."""

import math

import torch


def sample_hand_payloads(
    count: int,
    *,
    max_total_mass: float,
    empty_probability: float,
    left_fraction_range: tuple[float, float],
    com_offset: float,
    radius_range: tuple[float, float],
    difficulty: float = 1.0,
    min_total_mass: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample two wrist-attached load proxies, with one shared total-mass budget.

    Loaded totals are uniform in ``[min_total_mass, max_total_mass]``, scaled by ``difficulty``.
    COM offsets are in each wrist's link frame, inside a ball of ``com_offset``
    meters. Each proxy uses a solid-sphere inertia about its sampled center.
    This models distributed loads, not a shared rigid object or grasp contact.
    """
    scalars = (
        min_total_mass, max_total_mass, empty_probability, com_offset, difficulty, *left_fraction_range, *radius_range
    )
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("Payload parameters must be finite.")
    if (
        not 0 <= min_total_mass <= max_total_mass
        or com_offset < 0
        or not 0 <= empty_probability <= 1
        or not 0 <= difficulty <= 1
    ):
        raise ValueError("Invalid payload mass, offset, empty probability, or difficulty.")
    if not 0 < left_fraction_range[0] <= left_fraction_range[1] < 1:
        raise ValueError("Both hands must carry a positive fraction of the total load.")
    if not 0 < radius_range[0] <= radius_range[1]:
        raise ValueError("Payload radii must be positive and ordered.")
    total = (min_total_mass + torch.rand(count) * (max_total_mass - min_total_mass)) * difficulty
    total[torch.rand(count) < empty_probability] = 0.0
    left = torch.empty(count).uniform_(*left_fraction_range)
    masses = total[:, None] * torch.stack((left, 1.0 - left), dim=-1)
    direction = torch.randn(count, 2, 3)
    direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    centers = direction * torch.rand(count, 2, 1).pow(1.0 / 3.0) * com_offset * difficulty
    radii = torch.empty(count, 2).uniform_(*radius_range)
    inertia = (0.4 * masses * radii.square())[..., None, None] * torch.eye(3)
    return masses, centers, inertia


def combine_payload_inertia(
    link_mass: torch.Tensor,
    link_com: torch.Tensor,
    link_inertia: torch.Tensor,
    payload_mass: torch.Tensor,
    payload_com: torch.Tensor,
    payload_inertia: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine two rigid bodies using the parallel-axis theorem.

    Centers and inertia axes are expressed in the same link frame. Each input
    inertia is about its own center of mass; the result is about the combined
    center. Leading dimensions are batches (for example environments and hands).
    Always pass the original link properties to avoid accumulating old payloads.
    """
    mass = link_mass + payload_mass
    center = (link_mass[..., None] * link_com + payload_mass[..., None] * payload_com) / mass[..., None]
    eye = torch.eye(3, dtype=link_inertia.dtype, device=link_inertia.device)

    def shift(weight, displacement):
        outer = displacement[..., :, None] * displacement[..., None, :]
        return weight[..., None, None] * (displacement.square().sum(-1)[..., None, None] * eye - outer)

    inertia = (
        link_inertia
        + payload_inertia
        + shift(link_mass, link_com - center)
        + shift(payload_mass, payload_com - center)
    )
    return mass, center, inertia
