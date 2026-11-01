"""Payload inertia checks that do not require Isaac Sim."""

import importlib.util
from pathlib import Path

import torch
import pytest


_PATH = Path(__file__).resolve().parents[1] / "source/cyclotron/cyclotron/tasks/locomotion/mdp/payload.py"
_SPEC = importlib.util.spec_from_file_location("payload", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
combine_payload_inertia = _MODULE.combine_payload_inertia
sample_hand_payloads = _MODULE.sample_hand_payloads


def draw_payloads(count=10_000, **overrides):
    params = dict(
        max_total_mass=5.0, empty_probability=0.3, left_fraction_range=(0.4, 0.6),
        com_offset=0.05, radius_range=(0.04, 0.08),
    )
    return sample_hand_payloads(count, **(params | overrides))


def test_total_load_budget_empty_fraction_and_two_hand_split():
    torch.manual_seed(42)
    mass, center, inertia = draw_payloads()
    total = mass.sum(dim=-1)
    loaded = total > 0
    assert 0.28 < (~loaded).float().mean() < 0.32
    assert total.max() <= 5.0
    assert total.max() > 4.9
    fraction = mass[loaded, 0] / total[loaded]
    assert ((fraction >= 0.4) & (fraction <= 0.6)).all()
    assert center.norm(dim=-1).max() <= 0.05 + 1e-6
    assert torch.count_nonzero(inertia[~loaded]) == 0
    assert torch.linalg.eigvalsh(inertia[loaded]).min() > 0


def test_empty_and_curriculum_endpoints():
    assert torch.count_nonzero(draw_payloads(10, empty_probability=1.0)[0]) == 0
    assert torch.count_nonzero(draw_payloads(10, difficulty=0.0)[0]) == 0
    mass, center, _ = draw_payloads(100, difficulty=0.2, empty_probability=0.0)
    assert mass.sum(dim=-1).max() <= 1.0
    assert center.norm(dim=-1).max() <= 0.01 + 1e-6


def test_minimum_total_load():
    torch.manual_seed(0)
    total = draw_payloads(min_total_mass=5.0, max_total_mass=8.0)[0].sum(dim=-1)
    loaded = total[total > 0]
    assert 0.28 < 1 - len(loaded) / len(total) < 0.32
    assert loaded.min() >= 5.0 - 1e-5 and loaded.max() <= 8.0 + 1e-5
    assert loaded.min() < 5.1 and loaded.max() > 7.9
    scaled = draw_payloads(min_total_mass=5.0, max_total_mass=8.0, difficulty=0.2, empty_probability=0.0)[0]
    assert scaled.sum(dim=-1).min() >= 1.0 - 1e-5 and scaled.sum(dim=-1).max() <= 1.6 + 1e-5


@pytest.mark.parametrize("overrides", [
    {"max_total_mass": -1}, {"min_total_mass": 6.0}, {"min_total_mass": -1.0}, {"empty_probability": 2}, {"left_fraction_range": (0, 1)},
    {"radius_range": (0.0, 0.1)}, {"com_offset": -0.1}, {"difficulty": float("nan")},
])
def test_invalid_payload_parameters(overrides):
    with pytest.raises(ValueError):
        draw_payloads(1, **overrides)


def test_offset_payload_has_expected_center_and_parallel_axis_inertia():
    mass, center, inertia = combine_payload_inertia(
        torch.tensor([2.0]), torch.zeros(1, 3), 0.2 * torch.eye(3).unsqueeze(0),
        torch.tensor([1.0]), torch.tensor([[1.0, 0.0, 0.0]]), torch.zeros(1, 3, 3),
    )
    torch.testing.assert_close(mass, torch.tensor([3.0]))
    torch.testing.assert_close(center, torch.tensor([[1.0 / 3.0, 0.0, 0.0]]))
    torch.testing.assert_close(inertia[0], torch.diag(torch.tensor([0.2, 0.2 + 2.0 / 3.0, 0.2 + 2.0 / 3.0])))


def test_zero_payload_preserves_original_properties_without_accumulation():
    mass = torch.tensor([[0.4, 0.6], [0.5, 0.7]])
    center = torch.randn(2, 2, 3)
    inertia = torch.eye(3).expand(2, 2, 3, 3).clone()
    for _ in range(2):
        actual = combine_payload_inertia(
            mass, center, inertia, torch.zeros_like(mass), torch.randn_like(center), torch.zeros_like(inertia)
        )
        for result, expected in zip(actual, (mass, center, inertia)):
            torch.testing.assert_close(result, expected)


def test_combined_inertia_is_rotation_equivariant_and_positive_definite():
    rotation, _ = torch.linalg.qr(torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0]]))
    m1, m2 = torch.tensor([0.5]), torch.tensor([2.0])
    c1, c2 = torch.tensor([[0.1, -0.2, 0.0]]), torch.tensor([[0.3, 0.4, 0.1]])
    i1 = torch.diag(torch.tensor([0.01, 0.02, 0.03])).unsqueeze(0)
    i2 = torch.diag(torch.tensor([0.03, 0.04, 0.05])).unsqueeze(0)
    _, center, inertia = combine_payload_inertia(m1, c1, i1, m2, c2, i2)
    _, rotated_center, rotated_inertia = combine_payload_inertia(
        m1, c1 @ rotation.T, rotation @ i1 @ rotation.T,
        m2, c2 @ rotation.T, rotation @ i2 @ rotation.T,
    )
    torch.testing.assert_close(rotated_center, center @ rotation.T)
    torch.testing.assert_close(rotated_inertia, rotation @ inertia @ rotation.T)
    assert torch.linalg.eigvalsh(inertia).min() > 0
