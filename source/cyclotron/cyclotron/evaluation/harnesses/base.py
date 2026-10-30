"""Small explicit adapter registry: datasets never supply executable plugins."""

import os


def select_harness(root, explicit=None, benchmark=None):
    from .hpv import HPV

    registry = {HPV.name: HPV}
    aliases = {HPV.repo: HPV.name}
    harness = (benchmark or {}).get("harness") or {}
    if not isinstance(harness, dict):
        raise ValueError("benchmark.yaml's harness must be a mapping")
    declared = harness.get("name")
    repository = harness.get("repo") or ""
    if not isinstance(repository, str) or (declared is not None and not isinstance(declared, str)):
        raise ValueError("benchmark.yaml's harness.name and harness.repo must be strings")
    repository = repository.removesuffix(".git").rstrip("/")
    by_repo = aliases.get(repository)
    if repository and not by_repo:
        raise ValueError(f"No installed adapter for harness repository {repository}")
    if declared and by_repo and declared != by_repo:
        raise ValueError("harness.name conflicts with harness.repo")
    expected = declared or by_repo
    selected = explicit or os.environ.get("CYCLOTRON_HARNESS")
    if selected and expected and selected != expected:
        raise ValueError(f"Requested harness {selected} conflicts with benchmark harness {expected}")
    name = selected or expected or HPV.name
    if name not in registry:
        raise ValueError(f"Unknown harness {name!r}; available: {', '.join(registry)}")
    return registry[name](root)
