"""Prepare a training run for sharing on the Hugging Face Hub with ``./cyclotron.sh --share``.

Plain Python with no Isaac Lab imports, so it can run without Isaac Sim and in tests.
"""

from __future__ import annotations

import ntpath
import re

# Keys holding files on the training machine, e.g. asset_path, cache_dir, motion_files. USD prim paths (prim_path,
# filter_prim_paths_expr, ...) also start with "/" but name scene objects, not files, so they are kept.
_LOCAL_PATH_KEY = re.compile(r"\w*(_path|_dir|_file|_files)")
_ABSOLUTE_PATH = re.compile(r"(/|~|[A-Za-z]:[\\/])")
_KEY_LINE = re.compile(r"( *)(- )?([\w.-]+): ?(.*?)(\s*)$")
_ITEM_LINE = re.compile(r"( *)- (.*?)(\s*)$")


def _strip_value(value: str, removed: list[str]) -> str:
    quote = value[0] if value[:1] in ("'", '"') and value[-1:] == value[:1] else ""
    path = value[1:-1] if quote else value
    if not _ABSOLUTE_PATH.match(path):
        return value
    removed.append(path)
    return quote + ntpath.basename(path.rstrip("/\\")) + quote


def strip_local_paths(text: str) -> tuple[str, list[str]]:
    """Replace training-machine file paths in a run's yaml with their file names, so the yaml can be shared.

    Works line by line rather than loading and dumping the yaml, so every other line, including Isaac Lab's
    Python tags, stays byte for byte the same. Returns the new text and the paths that were replaced.
    """
    lines, removed = [], []
    list_indent = None  # indent of a key like motion_files whose path list is being read
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        newline = line[len(body) :]
        item = _ITEM_LINE.fullmatch(body)
        if list_indent is not None and item and len(item.group(1)) >= list_indent and ": " not in item.group(2):
            lines.append(item.group(1) + "- " + _strip_value(item.group(2), removed) + item.group(3) + newline)
            continue
        list_indent = None
        key = _KEY_LINE.fullmatch(body)
        if key and _LOCAL_PATH_KEY.fullmatch(key.group(3)) and "prim" not in key.group(3):
            indent, dash, name, value, trailing = key.groups()
            if not value:
                list_indent = len(indent) + len(dash or "")
            else:
                body = f"{indent}{dash or ''}{name}: {_strip_value(value, removed)}{trailing}"
        lines.append(body + newline)
    return "".join(lines), removed
