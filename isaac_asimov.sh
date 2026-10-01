#!/usr/bin/env bash

set -euo pipefail

ISAAC_ASIMOV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PYTHON_EXE="${PYTHON_EXE:-python}"

usage() {
    echo "Usage: $0 {--install|--list|--train|--play|--share|--view} [arguments]"
}

case "${1:-}" in
    -i|--install)
        "${PYTHON_EXE}" -m pip install -e "${ISAAC_ASIMOV_ROOT}/source/isaac_asimov"
        ;;
    -l|--list)
        shift
        "${PYTHON_EXE}" "${ISAAC_ASIMOV_ROOT}/scripts/list_envs.py" "$@"
        ;;
    -t|--train)
        shift
        "${PYTHON_EXE}" "${ISAAC_ASIMOV_ROOT}/scripts/rsl_rl/train.py" "$@"
        ;;
    -p|--play)
        shift
        "${PYTHON_EXE}" "${ISAAC_ASIMOV_ROOT}/scripts/rsl_rl/play.py" "$@"
        ;;
    -s|--share)
        shift
        "${PYTHON_EXE}" "${ISAAC_ASIMOV_ROOT}/scripts/share.py" "$@"
        ;;
    -v|--view)
        shift
        VIEWER_DIR="${ISAAC_ASIMOV_ROOT}/third_party/humanoid-policy-viewer"
        if ! command -v node >/dev/null 2>&1; then
            echo "[ERROR] --view needs Node.js 20 or newer: https://nodejs.org" >&2
            exit 1
        fi
        if [ ! -f "${VIEWER_DIR}/package.json" ]; then
            echo "Fetching humanoid-policy-viewer (first run only)..."
            git -C "${ISAAC_ASIMOV_ROOT}" submodule update --init --checkout --depth 1 third_party/humanoid-policy-viewer
        fi
        # Without a display there is no browser to open; the viewer prints its URL instead.
        if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
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
