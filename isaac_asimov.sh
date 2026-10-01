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
        "${PYTHON_EXE}" "${ISAAC_ASIMOV_ROOT}/scripts/rsl_rl/view.py" "$@"
        ;;
    *)
        usage
        exit 2
        ;;
esac
