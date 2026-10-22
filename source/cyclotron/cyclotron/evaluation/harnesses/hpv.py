"""HPV runs local inputs; all Hugging Face operations belong to Python."""

import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

from ..hub import cache_root
from ..provenance import file_manifest, git_state

INSPECTION_SCHEMAS = {"benchmark": "hpv-inspection/1", "policy": "hpv-policy-inspection/1"}


class HPV:
    name = "humanoid-policy-viewer"
    repo = "https://github.com/menloresearch/humanoid-policy-viewer"
    dataset_patterns = (
        r"benchmark\.yaml", r"[A-Z_]+\.md", r"data/[^/]+/test\.jsonl", r"calibration/[^/]+\.json",
        r"tools/[^/]+\.(mjs|js|py)", r"(LICEN[CS]E|COPYING)(\.[A-Za-z0-9]+)?",
    )

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.path = self.root / "third_party" / self.name
        self.results = self.root / "logs" / "benchmarks" / self.name
        self.workspaces = self.root / "logs" / "benchmark_workspaces" / self.name
        self._prepared = False

    @staticmethod
    def model_files(names):
        metadata = {prefix + name for prefix in ("", "params/")
                    for name in ("env.yaml", "agent.yaml", "tracking_policy.json")}
        onnx = [name for name in names if name.lower().endswith(".onnx")]
        if not onnx:
            raise ValueError("Model repository contains no ONNX policy")
        return [name for name in names if name in metadata or name in onnx or name.endswith(".onnx.data")]

    @classmethod
    def dataset_files(cls, names):
        return [name for name in names if name == "eval.yaml" or any(re.fullmatch(p, name) for p in cls.dataset_patterns)]

    def state(self):
        return {"name": self.name, "repo": self.repo, "pin": self.pin(), **self.checkout()}

    def checkout(self):
        # Untracked files cannot change a tracked module, and HPV's viewer writes an untracked
        # logs/; like HPV's own provenance, they do not make its checkout dirty.
        return git_state(self.path, untracked=False)

    def pin(self):
        line = subprocess.check_output(
            ["git", "-C", str(self.root), "ls-files", "--stage", "third_party/" + self.name], text=True
        ).strip()
        if not line:
            raise ValueError("HPV must be a pinned cyclotron submodule")
        return line.split()[1]

    def prepare(self):
        if self._prepared:
            return
        try:
            version = subprocess.check_output(["node", "-p", "process.versions.node"], text=True).strip()
        except FileNotFoundError:
            version = "0"
        if int(version.split(".")[0]) < 20:
            raise ValueError(f"HPV needs Node.js 20 or newer (found {version}). Install it from https://nodejs.org")
        submodule = "third_party/" + self.name
        refresh = False
        if not (self.path / "package.json").is_file():
            subprocess.run(["git", "-C", str(self.root), "submodule", "update", "--init", "--checkout", "--depth", "1", submodule], check=True)
        elif self.checkout()["commit"] != self.pin():
            if self.checkout()["dirty"]:
                print(f"Warning: {submodule} has local changes; leaving its checkout untouched", file=sys.stderr)
            else:
                subprocess.run(["git", "-C", str(self.root), "submodule", "update", "--checkout", "--depth", "1", submodule], check=True)
                refresh = True
        if refresh or not (self.path / "node_modules/vite/package.json").is_file():
            subprocess.run(["npm", "ci"], cwd=self.path, check=True)
        self._prepared = True

    def env(self):
        from ..runner import DEFAULT_BENCHMARK

        env = os.environ.copy()
        owned = cache_root() / "harnesses" / self.name
        temporary = owned / "tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        # HPV's "New benchmark" starts from cyclotron's own default suite, and nothing HPV
        # writes lands in its checkout.
        env.update(HPV_CACHE_DIR=str(owned), HPV_POLICY_LIBRARY_DIR=str(owned / "policies"),
                   HPV_POLICY_ROOTS="policies", HPV_BENCHMARK_RESULTS_DIR=str(self.results),
                   HPV_BENCHMARKS_DIR=str(self.workspaces), HPV_BENCHMARK_TEMPLATE=str(self.root / DEFAULT_BENCHMARK),
                   HPV_LOG_DIR=str(owned / "logs"), TMPDIR=str(temporary))
        # No harness operation should ever need Hub credentials.
        env.pop("HF_TOKEN", None)
        env.pop("HUGGING_FACE_HUB_TOKEN", None)
        return env

    def execute(self, args, capture=False):
        self.prepare()
        command = ["node", *map(str, args)]
        if capture:
            completed = subprocess.run(command, cwd=self.path, env=self.env(), text=True, capture_output=True)
            if completed.returncode:
                raise ValueError(completed.stderr.strip() or completed.stdout.strip())
            return json.loads(completed.stdout)
        # Piped or logged output is block-buffered; print what we said before the harness does.
        sys.stdout.flush()
        sys.stderr.flush()
        process = subprocess.Popen(command, cwd=self.path, env=self.env())
        previous, received = {}, []
        def forward(signum, frame):
            received.append(signum)
            # Ctrl+C already reaches the harness through the terminal's process group, and a
            # second SIGINT would cut its graceful shutdown short; forward only SIGTERM.
            if signum != signal.SIGINT and process.poll() is None:
                process.send_signal(signum)
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, forward)
            status = process.wait()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        if received:
            # A deliberate stop, not a harness failure: exit quietly with 130.
            raise KeyboardInterrupt
        if status:
            raise subprocess.CalledProcessError(status, command)

    @staticmethod
    def headless_options(options):
        words = list(options)
        if os.environ.get("SSH_CONNECTION") or (sys.platform != "darwin" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))):
            if "no-open" not in words:
                words.append("no-open")
        return words

    def view(self, policies, benchmark, options):
        # HPV lists every policy, labelled by its shortest distinguishing path, and selects the first.
        words = [f"policy={policy}" for policy in policies]
        self.execute(["scripts/hf.mjs", *words, f"benchmark={benchmark}", *self.headless_options(options)])

    def run(self, policies, benchmark, output, options):
        self.execute(["scripts/benchmark.mjs", *policies, benchmark, f"out={output}", *options])

    def view_result(self, directory, options):
        self.execute(["scripts/benchmark.mjs", "view", directory, *self.headless_options(options)])

    def inspect(self, kind, *args):
        # HPV's offline inspect command: its own validation and hashes, no network or Hub access.
        result = self.execute(["scripts/inspect.mjs", kind, *args], capture=True)
        if result.get("schema") != INSPECTION_SCHEMAS[kind]:
            raise ValueError(f"Unsupported HPV {kind} inspection schema: {result.get('schema')}")
        return result

    def inspect_benchmark(self, directory, commit=None):
        return self.inspect("benchmark", directory, *([commit] if commit else []))

    def task_specs(self, inspection):
        return inspection["tasks"]

    def task_harness(self, inspection, commit):
        # Each eval.yaml task names the harness that runs and scores it, under the benchmark's protocol.
        protocol = (inspection["raw"].get("harness") or {}).get("protocol")
        return {"name": self.name, "repo": self.repo, "protocol": protocol, "commit": commit}

    def inspect_policy(self, item):
        info = self.inspect("policy", item["path"])
        if info["errors"]:
            raise ValueError(f"{item['path']} cannot be run:\n" + "\n".join(f"  - {error}" for error in info["errors"]))
        directory = Path(info["policyDir"])
        return {**item, "path": str(directory), "primary": info["primary"], "hpv_hashes": info["hashes"],
                "files": file_manifest(directory)}

    def normalize_result(self, result):
        if result.get("schema") != "hpv-benchmark-result/1":
            raise ValueError("Unsupported HPV result schema")
        return result

    def binding_errors(self, run, manifest):
        reasons = []
        inspection = manifest["inspection"]
        policies = manifest["policies"]
        if len(run.get("policies", [])) != len(policies):
            reasons.append("Result policies do not match recorded inputs")
        for index, (native, recorded) in enumerate(zip(run.get("policies", []), policies)):
            if native.get("key") != f"m{index}" or native.get("primary") != recorded["primary"] or native.get("files") != recorded["hpv_hashes"]:
                reasons.append(f"Policy m{index} does not match the recorded ONNX/config hashes")
        if run.get("benchmark", {}).get("raw") != inspection["raw"]:
            reasons.append("Result benchmark settings do not match the resolved input")
        native_harness = run.get("provenance", {}).get("hpv", {})
        if native_harness.get("commit") != manifest["harness"]["commit"]:
            reasons.append("Result harness commit differs from the recorded checkout")
        return reasons

    def submit_blockers(self, run, manifest):
        reasons = self.binding_errors(run, manifest)
        inspection = manifest["inspection"]
        provenance = run.get("provenance", {}).get("hpv", {})
        if provenance.get("dirty") is not False or not provenance.get("commit"):
            reasons.append("HPV reported a dirty or unknown checkout")
        if inspection["raw"].get("harness", {}).get("protocol") != provenance.get("protocol"):
            reasons.append("Benchmark protocol differs from the HPV runtime protocol")
        if inspection["errors"] or inspection["problems"] or run.get("benchmark", {}).get("problems"):
            reasons.append("Benchmark validation or compatibility checks failed")
        if any(run.get("overrides", {}).values()) or any(
            word == "force" or word.startswith(("tests=", "repeats=")) for word in manifest.get("options", [])
        ):
            reasons.append("Tests/repeats/force overrides make a run local-only")
        if run.get("shard"):
            reasons.append("Sharded runs are local-only; cyclotron does not yet finalize merged provenance")
        cells = run.get("cells", [])
        expected = {(f"m{index}", cell["cellId"]) for index in range(len(manifest["policies"])) for cell in inspection["cells"]}
        actual = [(cell.get("policyKey"), cell.get("cellId")) for cell in cells]
        if not expected or len(actual) != len(set(actual)) or set(actual) != expected or run.get("expectedCells") != len(expected):
            reasons.append("Result has missing, duplicate or unexpected cells")
        definitions = {c["cellId"]: c for c in inspection["cells"]}
        if any(any(cell.get(key) != definitions.get(cell.get("cellId"), {}).get(key) for key in ("testId", "repeat", "seed")) for cell in cells):
            reasons.append("Result cell seeds/repeats do not match the benchmark plan")
        if run.get("tests") != inspection["tests"]:
            reasons.append("Result test rows do not match the resolved benchmark")
        if any(cell.get("error") for cell in cells):
            reasons.append("One or more cells failed")
        expected_tasks = {task["id"] for task in inspection["tasks"]}
        for index in range(len(manifest["policies"])):
            tasks = run.get("tasks", {}).get(f"m{index}", [])
            if {task.get("taskId") for task in tasks} != expected_tasks or len(tasks) != len(expected_tasks):
                reasons.append(f"Policy m{index} has missing or duplicate task summaries")
        return reasons
