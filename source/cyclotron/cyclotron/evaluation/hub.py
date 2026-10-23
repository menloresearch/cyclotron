"""Cyclotron's Hub transport and immutable, real-file input cache."""

import glob
import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

from ..robot_model import cache_dir
from .provenance import file_names, write_json


def cache_root():
    return Path(cache_dir()).resolve()


def hub_api():
    from huggingface_hub import HfApi

    return HfApi(endpoint=os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/"))


REPO_ID = r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*"


def check_repo_id(repo_id):
    if not isinstance(repo_id, str) or not re.fullmatch(REPO_ID, repo_id) or ".." in repo_id or "--" in repo_id:
        raise ValueError(f"Invalid repository ID: {repo_id}")
    return repo_id


def parse_reference(text, repo_type="model", revision=None):
    """Parse an explicit Hub input; local-path detection happens before this."""
    if "://" in text:
        url = urlparse(text)
        endpoint = urlparse(os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
        if url.scheme not in ("https", "http") or url.netloc not in {endpoint.netloc, "huggingface.co", "hf.co"}:
            raise ValueError(f"Not a configured Hugging Face URL: {text}")
        if url.query or url.fragment:
            raise ValueError("Use a repository URL without a query or fragment")
        text = unquote(url.path.strip("/"))
        parts = text.split("/tree/", 1)
        if len(parts) == 2:
            text = f"{parts[0]}@{parts[1]}"
    if text.startswith("datasets/"):
        repo_type, text = "dataset", text[len("datasets/"):]
    repo_id, separator, embedded = text.partition("@")
    if not re.fullmatch(REPO_ID, repo_id):
        raise ValueError(f"Expected an existing local path or org/repo[@revision], got {text!r}")
    check_repo_id(repo_id)
    if separator and (not embedded or (revision and revision != embedded)):
        raise ValueError("Empty or conflicting revisions")
    return {"id": repo_id, "repo_type": repo_type, "revision": embedded or revision or "main"}


def is_dataset(text):
    return text.startswith("datasets/") or "/datasets/" in text and "://" in text


def local_path(text, policy=False):
    path = Path(text).expanduser()
    if path.exists():
        if policy and path.is_dir() and (path / "exported").is_dir() and not list(path.glob("*.onnx")):
            path /= "exported"
        return path.resolve()
    if text.startswith((".", "/", "~")) or text.lower().endswith(".onnx"):
        raise ValueError(f"Local path does not exist: {text}")
    return None


def safe_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or any(p.startswith(".") for p in path.parts):
        raise ValueError(f"Unsafe Hub filename: {name}")
    return name


def _verify_blob(path, sibling):
    """Hub LFS hashes are SHA256; ordinary files have a Git blob SHA1. Returns the file's SHA256,
    read in the same pass, for the manifest."""
    lfs = getattr(sibling, "lfs", None)
    expected = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
    digest = hashlib.sha256()
    blob = None if expected else hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            if blob:
                blob.update(chunk)
    if blob:
        expected = getattr(sibling, "blob_id", None)
    actual = blob.hexdigest() if blob else digest.hexdigest()
    if not expected or actual != expected:
        raise ValueError(f"Cannot verify pinned Hub file: {sibling.rfilename}")
    return digest.hexdigest()


def resolve_input(text, repo_type, select_files, revision=None, api=None):
    path = local_path(text, policy=repo_type == "model")
    if path is not None:
        return {"source": "local", "path": str(path), "repo_type": repo_type}
    reference = parse_reference(text, repo_type, revision)
    if reference["repo_type"] != repo_type:
        raise ValueError(f"Expected a {repo_type}, got {text}")
    return download(reference, select_files, api=api)


def download(reference, select_files, api=None):
    from filelock import FileLock
    from huggingface_hub import snapshot_download

    api = api or hub_api()
    info = api.repo_info(reference["id"], repo_type=reference["repo_type"], revision=reference["revision"], files_metadata=True)
    sha = info.sha
    if not re.fullmatch(r"[a-fA-F0-9]{40,64}", sha or ""):
        raise ValueError("Hub did not return a pinned commit")
    # Record the Hub's own spelling of the repo (its case, or its new name after a rename), not the user's.
    reference = {**reference, "id": check_repo_id(info.id)}
    siblings = {s.rfilename: s for s in info.siblings}
    names = sorted(safe_name(n) for n in select_files(list(siblings)))
    if not names:
        raise ValueError(f"No supported input files in {reference['id']}")
    endpoint = api.endpoint.rstrip("/")
    endpoint_key = hashlib.sha256(endpoint.encode()).hexdigest()[:16]
    # Harnesses label a policy by its real folder name, so the leaf names the repo, not the commit.
    destination = (cache_root() / "hub" / endpoint_key / reference["repo_type"] / reference["id"] / sha
                   / reference["id"].replace("/", "__"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock"):
        # Check bytes against the pinned remote metadata even when reusing the cache.
        files = {}
        if destination.exists():
            if not (destination / ".cyclotron-cache.json").is_file():
                raise ValueError(f"Incomplete cache: {destination}; move it aside and retry")
            for name in names:
                path = destination / name
                if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(destination):
                    raise ValueError(f"Invalid cached input: {path}; move the cache folder aside and retry")
                files[name] = _verify_blob(path, siblings[name])
            if set(file_names(destination)) != set(names):
                raise ValueError(f"Unexpected cached files: {destination}; move it aside and retry")
        else:
            with tempfile.TemporaryDirectory(prefix=".download-", dir=destination.parent) as temporary:
                staging = Path(temporary) / "files"
                source = Path(snapshot_download(
                    repo_id=reference["id"], repo_type=reference["repo_type"], revision=sha,
                    endpoint=endpoint, token=api.token, allow_patterns=[glob.escape(name) for name in names],
                    cache_dir=cache_root() / "hf-downloads" / endpoint_key,
                ))
                for name in names:
                    target = staging / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / name, target)
                    files[name] = _verify_blob(target, siblings[name])
                write_json(staging / ".cyclotron-cache.json", {"sha": sha, "files": files})
                staging.rename(destination)
    return {**reference, "source": "hub", "sha": sha, "endpoint": endpoint,
            "path": str(destination), "files": files}


def editable_copy(item, destination):
    """A folder to edit and publish from a pinned Hub benchmark; an existing folder is never touched."""
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise ValueError(f"{destination} already exists; give another folder, or move it aside to pull again")
    source = Path(item["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pull-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "files"
        # The cache's records stay behind (HPV's would make the copy read-only); publication builds eval.yaml again.
        shutil.copytree(source, staged, ignore=lambda folder, names: [
            name for name in names if name.startswith(".") or (Path(folder) == source and name == "eval.yaml")])
        write_json(staged / ".cyclotron-origin.json",
                   {"origin": {key: item[key] for key in ("id", "endpoint", "revision", "sha")}})
        staged.rename(destination)
    print(f"Pulled datasets/{item['id']}@{item['sha'][:12]} into {destination}")
    return destination


def on_main(api, repo_id, sha, repo_type="dataset"):
    """Whether a commit is in main's history: merged, not only a pending PR revision."""
    return sha in {commit.commit_id for commit in api.list_repo_commits(repo_id, repo_type=repo_type, revision="main")}


def read_repo_text(api, repo_id, path, revision, repo_type="model"):
    """A file's text at a revision, or None only when the Hub says the file does not exist there."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    try:
        endpoint_key = hashlib.sha256(api.endpoint.rstrip("/").encode()).hexdigest()[:16]
        downloaded = hf_hub_download(repo_id, path, repo_type=repo_type, revision=revision,
                                     endpoint=api.endpoint, token=api.token,
                                     cache_dir=cache_root() / "hf-downloads" / endpoint_key)
    except LocalEntryNotFoundError:
        # huggingface_hub reports connection errors, timeouts, 5xx, 429 and 403 this way: the
        # file may well exist, and treating it as absent would drop existing entries on commit.
        raise
    except EntryNotFoundError as error:
        # The remote "entry not found" (RemoteEntryNotFoundError in huggingface_hub 1.x).
        response = getattr(error, "response", None)
        if response is None or response.status_code != 404:
            raise
        return None
    return Path(downloaded).read_text()


def commit_files(api, repo_id, files, parent, message, repo_type="model", pr=False, deletions=()):
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    from huggingface_hub.errors import HfHubHTTPError

    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=content.encode() if isinstance(content, str) else content)
                  for name, content in files.items()]
    operations += [CommitOperationDelete(path_in_repo=name) for name in deletions]
    args = dict(repo_id=repo_id, repo_type=repo_type, operations=operations, parent_commit=parent, commit_message=message)
    # Ownership is not equivalent to write permission (organizations, fine-grained tokens).
    # Try the normal commit, falling back only on an explicit permission denial.
    try:
        return api.create_commit(**args, create_pr=pr)
    except HfHubHTTPError as error:
        if pr or error.response is None or error.response.status_code != 403:
            raise
        return api.create_commit(**args, create_pr=True)
