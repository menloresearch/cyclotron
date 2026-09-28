# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Build a labeled grid montage MP4 from the per-category clips `record_videos.py` writes.

Pure Python (cv2 + imageio) — no `isaaclab`/Kit app needed, so this runs fast and cheap after
`record_videos.py` has already closed the simulator and freed its VRAM/RAM.

Streams each source clip frame-by-frame (never loads a whole clip into memory) so the montage step
stays lightweight even with several ~600-frame 720p clips. Shorter clips freeze on their last frame
until the longest clip ends, so every tile spans the same duration.

The default grid is a fixed 4x2 at 1920x1080 (tiles 480x540). Each source clip is letterboxed (aspect
ratio preserved, black bars) into its tile instead of being squashed to a non-16:9 box, and each tile
gets a large, prominent category label. The clip's own overlay (run id / iter / assist / clock /
"standing at Xs", burned in by `record_videos.py`) is left as-is on the source frames, but it becomes
hard to read once a 1280x720 clip is shrunk that much, which the big category label compensates for.

Usage:
    python scripts/getup/make_montage.py --clips_dir ~/getup_results/<run>/videos/iter_999 \\
        --output ~/getup_results/<run>/videos/iter_999/montage.mp4
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys

import cv2
import imageio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as gc  # isort: skip

parser = argparse.ArgumentParser(description="Grid-montage the per-category get-up clips into one labeled MP4.")
parser.add_argument("--clips_dir", type=str, default=None, help="Directory of '<category>_ep<k>.mp4' files (as written by record_videos.py).")
parser.add_argument("--clips", type=str, nargs="*", default=None, help="Explicit 'label=path' pairs, used instead of --clips_dir.")
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--canvas_width", type=int, default=1920)
parser.add_argument("--canvas_height", type=int, default=1080)
parser.add_argument("--cols", type=int, default=4, help="Grid columns; rows = ceil(n_clips / cols). Tile size = canvas / (cols, rows).")
parser.add_argument("--episode_index", type=int, default=0, help="Which '_ep<k>' to pick per category, when scanning --clips_dir.")
args = parser.parse_args()


def _category_sort_key(label: str) -> tuple[int, str]:
    try:
        return (gc.CATEGORY_KEYS.index(label), label)
    except ValueError:
        return (len(gc.CATEGORY_KEYS), label)


def _discover_clips() -> list[tuple[str, str]]:
    if args.clips:
        pairs = []
        for item in args.clips:
            label, _, path = item.partition("=")
            pairs.append((label, path))
        return sorted(pairs, key=lambda kv: _category_sort_key(kv[0]))

    if not args.clips_dir:
        raise SystemExit("Pass either --clips_dir or --clips label=path ...")

    pattern = re.compile(rf"^(?P<label>.+)_ep{args.episode_index}\.mp4$")
    found: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(args.clips_dir, f"*_ep{args.episode_index}.mp4"))):
        m = pattern.match(os.path.basename(path))
        if m:
            found[m.group("label")] = path
    if not found:
        raise SystemExit(f"No '*_ep{args.episode_index}.mp4' clips found in {args.clips_dir}")
    return sorted(found.items(), key=lambda kv: _category_sort_key(kv[0]))


def _letterbox_fit(frame_rgb: np.ndarray, tile_w: int, tile_h: int) -> np.ndarray:
    """Resize `frame_rgb` to fit inside a `tile_w`x`tile_h` box, preserving aspect ratio, black-padded."""
    src_h, src_w = frame_rgb.shape[:2]
    scale = min(tile_w / src_w, tile_h / src_h)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    y0, x0 = (tile_h - new_h) // 2, (tile_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _label_tile(frame: np.ndarray, label: str, tile_w: int, tile_h: int) -> np.ndarray:
    """Burn a large, prominent category label into the top-left of the (already letterboxed) tile."""
    font_scale = max(0.9, tile_w / 480.0 * 1.1)
    thickness = max(2, round(font_scale * 1.8))
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    bar_h = text_h + baseline + 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (min(tile_w, text_w + 24), bar_h), (0, 0, 0), thickness=-1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    cv2.putText(frame, label, (12, bar_h - baseline - 6), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def main():
    clips = _discover_clips()
    n = len(clips)
    cols = max(1, min(args.cols, n))
    rows = max(1, math.ceil(n / cols))
    tw, th = args.canvas_width // cols, args.canvas_height // rows
    canvas_w, canvas_h = cols * tw, rows * th

    readers = []
    last_frames: list[np.ndarray | None] = []
    for label, path in clips:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise SystemExit(f"Could not open clip for category {label!r}: {path}")
        readers.append((label, cap))
        last_frames.append(None)

    writer = imageio.get_writer(args.output, fps=args.fps, codec="libx264", format="FFMPEG", macro_block_size=None, pixelformat="yuv420p")

    active = True
    n_written = 0
    while active:
        active = False
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        for idx, (label, cap) in enumerate(readers):
            ok, frame_bgr = cap.read()
            if ok:
                active = True
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                last_frames[idx] = _letterbox_fit(frame_rgb, tw, th)
            elif last_frames[idx] is None:
                last_frames[idx] = np.zeros((th, tw, 3), dtype=np.uint8)
            tile = _label_tile(last_frames[idx].copy(), label, tw, th)
            r, c = divmod(idx, cols)
            canvas[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = tile
        if n_written == 0 or active:
            writer.append_data(canvas)
            n_written += 1
        # Stop once every clip has frozen on its last frame for at least one extra written canvas frame.
        if not active and n_written > 0:
            break

    writer.close()
    for _, cap in readers:
        cap.release()
    print(f"[getup-eval] Wrote montage ({n} tiles, {cols}x{rows} grid @ {canvas_w}x{canvas_h}, {n_written} frames) to {args.output}")


if __name__ == "__main__":
    main()
