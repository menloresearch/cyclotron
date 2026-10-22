"""Small explicit adapter registry: datasets never supply executable plugins."""

import os
from typing import Protocol


class HarnessAdapter(Protocol):
    name: str
    repo: str

    def prepare(self): ...
    def view(self, policy, benchmark, options): ...
    def run(self, policies, benchmark, output, options): ...
    def view_result(self, directory, options): ...
    def inspect_benchmark(self, directory, commit=None): ...
    def normalize_result(self, result): ...
    def submit_blockers(self, result, manifest): ...


def select_app(root, explicit=None, benchmark=None):
    from .hpv import HPV

    registry = {HPV.name: HPV}
    aliases = {HPV.repo: HPV.name}
    harness = (benchmark or {}).get("harness", {})
    declared = harness.get("app")
    repository = harness.get("repo", "").removesuffix(".git").rstrip("/")
    by_repo = aliases.get(repository)
    if repository and not by_repo:
        raise ValueError(f"No installed adapter for harness repository {repository}")
    if declared and by_repo and declared != by_repo:
        raise ValueError("harness.app conflicts with harness.repo")
    selected = explicit or os.environ.get("CYCLOTRON_APP")
    expected = declared or by_repo
    if selected and expected and selected != expected:
        raise ValueError(f"Requested app {selected} conflicts with benchmark app {expected}")
    name = selected or expected or HPV.name
    if name not in registry:
        raise ValueError(f"Unknown app {name!r}; available: {', '.join(registry)}")
    return registry[name](root)
