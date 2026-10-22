#!/usr/bin/env bash

set -euo pipefail

CYCLOTRON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# An explicit PYTHON_EXE wins, then `python` (an active venv's or conda env's), then python3,
# which is all that many systems install.
if [ -z "${PYTHON_EXE:-}" ]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_EXE=python
    else
        PYTHON_EXE=python3
    fi
fi

usage() {
    echo "Usage: $0 {--install|--list|--train|--play|--export|--share|--view|--benchmark} [arguments]"
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
    -e|--export)
        shift
        "${PYTHON_EXE}" "${CYCLOTRON_ROOT}/scripts/rsl_rl/export.py" "$@"
        ;;
    -s|--share)
        shift
        "${PYTHON_EXE}" "${CYCLOTRON_ROOT}/scripts/share.py" "$@"
        ;;
    -v|--view)
        shift
        exec "${PYTHON_EXE}" "${CYCLOTRON_ROOT}/scripts/view.py" "$@"
        ;;
    -b|--benchmark)
        shift
        exec "${PYTHON_EXE}" "${CYCLOTRON_ROOT}/scripts/benchmark.py" "$@"
        ;;
    *)
        usage
        exit 2
        ;;
esac
