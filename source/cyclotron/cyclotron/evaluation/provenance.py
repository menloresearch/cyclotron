"""Reproducibility records, not attestations of trusted execution."""

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

SCHEMA = "cyclotron-evaluation/1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(root, names=None):
    root = Path(root).resolve()
    paths = (root / name for name in names) if names is not None else root.rglob("*")
    result = {}
    for path in sorted(paths):
        relative = path.relative_to(root)
        if names is None and any(part.startswith(".") or part == "node_modules" for part in relative.parts):
            continue
        if path.is_symlink() or (path.exists() and not path.resolve().is_relative_to(root)):
            raise ValueError(f"Input must not contain symlinks: {path}")
        if path.is_file():
            result[relative.as_posix()] = sha256(path)
        elif names is not None:
            raise ValueError(f"Missing input file: {path}")
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def git_state(path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL).strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def result_dir(path):
    path = Path(path).expanduser().resolve()
    return path.parent if path.name == "result.json" else path


def core_blockers(manifest, result_path):
    reasons = []
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        reasons.append("No complete, supported cyclotron provenance record")
    if manifest.get("result_sha256") != sha256(result_path):
        reasons.append("result.json changed after cyclotron finalized it")
    for name in ("cyclotron", "app"):
        state = manifest.get(name, {})
        if not state.get("commit") or state.get("dirty") is not False:
            reasons.append(f"The {name} checkout was dirty or its commit was unknown")
    inputs = [manifest.get("benchmark", {}), *manifest.get("policies", [])]
    if not manifest.get("policies"):
        reasons.append("No recorded policies")
    for item in inputs:
        if item.get("source") != "hub" or not item.get("sha") or not item.get("id"):
            reasons.append("Submission needs both a pinned Hub benchmark and pinned Hub policies")
            break
    if manifest.get("binding_errors"):
        reasons.extend(manifest["binding_errors"])
    return reasons
