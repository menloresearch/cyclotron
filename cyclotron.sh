#!/usr/bin/env bash

set -euo pipefail

CYCLOTRON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PYTHON_EXE="${PYTHON_EXE:-python}"

usage() {
    echo "Usage: $0 {--install|--list|--train|--play|--share|--view} [arguments]"
}

case "${1:-}" in
    -i|--install)
        "${PYTHON_EXE}" -m pip install -e "${CYCLOTRON_ROOT}/source/cyclotron"
        ;;
    -l|--list)
        shift
        "${PYTHON_EXE}" "${CYCLOTRON_ROOT}/scripts/list_envs.py" "$@"
        ;;
    -t|--train)
        shift
        "${PYTHON_EXE}" "${CYCLOTRON_ROOT}/scripts/rsl_rl/train.py" "$@"
        ;;
    -p|--play)
        shift
        "${PYTHON_EXE}" "${CYCLOTRON_ROOT}/scripts/rsl_rl/play.py" "$@"
        ;;
    -s|--share)
        shift
        "${PYTHON_EXE}" "${CYCLOTRON_ROOT}/scripts/share.py" "$@"
        ;;
    -v|--view)
        shift
        VIEWER_DIR="${CYCLOTRON_ROOT}/third_party/humanoid-policy-viewer"
        NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"
        if [ "${NODE_MAJOR}" -lt 20 ]; then
            echo "[ERROR] --view needs Node.js 20 or newer (found: $(node --version 2>/dev/null || echo none)): https://nodejs.org" >&2
            exit 1
        fi
        if [ ! -f "${VIEWER_DIR}/package.json" ]; then
            echo "Fetching humanoid-policy-viewer (first run only)..."
            git -C "${CYCLOTRON_ROOT}" submodule update --init --checkout --depth 1 third_party/humanoid-policy-viewer
        fi
        # Over SSH, or on Linux without a display, there is no browser to open; the viewer prints its URL instead.
        # macOS sets neither DISPLAY nor WAYLAND_DISPLAY but can always open one.
        if [ -n "${SSH_CONNECTION:-}" ] || { [ "$(uname)" != "Darwin" ] && [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; }; then
            set -- "$@" --no-open
        fi
        # Vite's vuetify plugin resolves packages from the working directory, so run from the viewer's checkout.
        cd "${VIEWER_DIR}"
        exec node scripts/run-hf-model.mjs "$@"
        ;;
    *)
        usage
        exit 2
        ;;
esac
