"""Thin script CLI: common flags translated, harness words passed through."""

import argparse
import sys
from pathlib import Path


VALUE_OPTIONS = ("harness", "revision", "benchmark", "policy", "out", "port", "tests", "repeats", "jobs", "robots", "shard")
FLAG_OPTIONS = ("no-open", "force", "watch", "verbose")


def parse_run(argv, viewing=False):
    parser = argparse.ArgumentParser(
        description="View policies, or every exported run in a folder, and edit a benchmark in the harness."
        if viewing else "Run a local or Hugging Face benchmark.",
        epilog="Known --flags and HPV key=value words are accepted. Unknown harness options pass through. "
               "Benchmark actions: view, submit, publish, release. No uploads occur during a run.",
        allow_abbrev=False,
    )
    parser.add_argument("targets", nargs="*", help="Policy paths/Hub IDs and optionally a benchmark folder or datasets/org/name")
    for name in VALUE_OPTIONS:
        parser.add_argument("--" + name, action="append")
    for name in FLAG_OPTIONS:
        parser.add_argument("--" + name, action="store_true")
    normalized, passthrough = [], []
    after_separator = False
    for word in argv:
        if word == "--":
            after_separator = True
            continue
        if "=" in word and not Path(word).exists():
            key, value = word.split("=", 1)
            key = key.removeprefix("--")
            if key in VALUE_OPTIONS:
                normalized.extend(("--" + key, value))
            else:
                passthrough.append(word)
        elif word in FLAG_OPTIONS:
            normalized.append("--" + word)
        elif after_separator and not word.startswith("--"):
            # A bare positional here could bypass input resolution and trigger HPV HF access.
            raise ValueError(f"Passthrough positionals are not supported: {word}; give policies before --")
        else:
            normalized.append(word)
    args, unknown = parser.parse_known_intermixed_args(normalized)
    passthrough += unknown
    for name in VALUE_OPTIONS:
        values = getattr(args, name)
        if name != "policy":
            if values and len(values) > 1:
                raise ValueError(f"Give --{name} only once (including its key=value alias)")
            setattr(args, name, values[0] if values else None)
    from .hub import is_dataset, local_path

    policies = []
    benchmark = args.benchmark
    for word in args.targets:
        path = local_path(word)
        if is_dataset(word) or (path and path.is_dir() and (path / "benchmark.yaml").is_file()):
            if benchmark:
                raise ValueError("Give exactly one benchmark")
            benchmark = word
        else:
            policies.append(word)
    policies.extend(args.policy or [])
    if not policies:
        parser.error(f"Give at least one policy to {'view' if viewing else 'benchmark'}")
    if viewing and (args.out or any(getattr(args, name) for name in ("tests", "repeats", "jobs", "robots", "shard", "force", "watch"))):
        raise ValueError("Run options belong to --benchmark, not --view")
    options = [f"{name}={getattr(args, name)}" for name in ("port", "tests", "repeats", "jobs", "robots", "shard") if getattr(args, name)]
    options += [name for name in FLAG_OPTIONS if getattr(args, name.replace("-", "_"))]
    options += passthrough
    return args, policies, benchmark, options


def main(root, argv=None, viewing=False):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = ["--help"]
    action = None if viewing or argv[0] not in ("view", "submit", "publish", "release") else argv.pop(0)
    if action:
        parser = argparse.ArgumentParser(description=f"Cyclotron benchmark {action}", allow_abbrev=False)
        if action in ("view", "submit"):
            parser.add_argument("result")
        if action in ("submit", "publish"):
            parser.add_argument("--pr", action="store_true")
        if action in ("submit", "publish", "release"):
            parser.add_argument("--dry-run", action="store_true")
        if action == "view":
            parser.add_argument("--port")
            parser.add_argument("--no-open", action="store_true")
        if action == "publish":
            parser.add_argument("directory")
            parser.add_argument("dataset")
            parser.add_argument("--base")
            parser.add_argument("--create", action="store_true")
            parser.add_argument("--harness")
        if action == "release":
            parser.add_argument("dataset")
            parser.add_argument("revision")
        args = parser.parse_args(argv)
        if action == "view":
            from .harnesses.base import select_harness
            from .provenance import read_json, result_dir

            directory = result_dir(args.result)
            native = read_json(directory / "result.json")
            sidecar = directory / "cyclotron.json"
            harness_name = read_json(sidecar)["harness"]["name"] if sidecar.is_file() else None
            harness = select_harness(root, harness_name, native.get("benchmark", {}).get("raw"))
            options = ([f"port={args.port}"] if args.port else []) + (["no-open"] if args.no_open else [])
            return harness.view_result(directory, options)
        if action == "submit":
            from .eval_results import submit

            return submit(root, args.result, args.pr, args.dry_run)
        from .publishing import publish, release

        if action == "publish":
            result = publish(root, args.directory, args.dataset, args.base, args.create, args.pr, args.dry_run, args.harness)
            if args.dry_run and result["blockers"]:
                # Everything is printed first; then fail, as the publication itself would.
                raise SystemExit(1)
            return result
        return release(root, args.dataset, args.revision, args.dry_run)
    args, policies, benchmark, options = parse_run(argv, viewing)
    from .runner import run, view

    if viewing:
        return view(root, policies, benchmark, args.harness, args.revision, options)
    return run(root, policies, benchmark, args.harness, args.revision, args.out, options)


def entry(root, viewing=False):
    import subprocess

    try:
        main(root, viewing=viewing)
    except ImportError as error:
        print(f"[ERROR] Missing evaluation dependency ({error}). Install with:\n"
              f"  {sys.executable} -m pip install -r {root / 'requirements-eval.txt'}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except subprocess.CalledProcessError as error:
        print(f"[ERROR] Harness command exited with status {error.returncode}", file=sys.stderr)
        return error.returncode if error.returncode > 0 else 128 - error.returncode
    except (ValueError, OSError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    return 0
