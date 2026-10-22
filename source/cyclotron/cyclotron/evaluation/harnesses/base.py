"""Small explicit adapter registry: datasets never supply executable plugins."""

import os
from typing import Protocol


class HarnessAdapter(Protocol):
    name: str
    repo: str

    def prepare(self): ...
    def view(self, policies, benchmark, options): ...
    def run(self, policies, benchmark, output, options): ...
    def view_result(self, directory, options): ...
    def inspect_benchmark(self, directory, commit=None): ...
    def task_harness(self, inspection, commit): ...
    def normalize_result(self, result): ...
    def submit_blockers(self, result, manifest): ...


def eval_harness(spec):
    """The harness eval.yaml's tasks name in their harness block; None for benchmarks published without one."""
    blocks = [task["harness"] for task in (spec or {}).get("tasks", [])
              if isinstance(task, dict) and isinstance(task.get("harness"), dict)]
    names = {block.get("name") for block in blocks}
    if None in names or len(names) > 1:
        raise ValueError(f"eval.yaml tasks must name one harness, not {sorted(map(str, names))}")
    return next(iter(names), None)


def select_harness(root, explicit=None, benchmark=None, spec=None):
    from .hpv import HPV

    registry = {HPV.name: HPV}
    aliases = {HPV.repo: HPV.name}
    harness = (benchmark or {}).get("harness", {})
    declared = harness.get("name")
    repository = harness.get("repo", "").removesuffix(".git").rstrip("/")
    by_repo = aliases.get(repository)
    if repository and not by_repo:
        raise ValueError(f"No installed adapter for harness repository {repository}")
    if declared and by_repo and declared != by_repo:
        raise ValueError("harness.name conflicts with harness.repo")
    expected = declared or by_repo
    published = eval_harness(spec)
    if published and expected and published != expected:
        raise ValueError(f"eval.yaml's harness {published} conflicts with benchmark.yaml's {expected}")
    expected = published or expected
    selected = explicit or os.environ.get("CYCLOTRON_HARNESS")
    if selected and expected and selected != expected:
        raise ValueError(f"Requested harness {selected} conflicts with benchmark harness {expected}")
    name = selected or expected or HPV.name
    if name not in registry:
        raise ValueError(f"Unknown harness {name!r}; available: {', '.join(registry)}")
    return registry[name](root)
