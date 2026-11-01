#!/usr/bin/env bash
# One distributed run of the 27-DOF blind lower-body AMP task with a 5-8 kg two-hand payload:
# four workers, 4,096 environments per worker (16,384 total), 0-25 ms actuator delay.
set -euo pipefail

CYCLOTRON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "${CYCLOTRON_ROOT}"
PYTHON_EXE="${PYTHON_EXE:-python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
# On the RTX PRO 6000 Blackwell host with NCCL 2.26.2, NCCL's default protocol hits an
# illegal memory access (kernel Xid 13) in the first broadcast; the LL protocol works.
# https://github.com/NVIDIA/nccl/issues/2418
export NCCL_PROTO="${NCCL_PROTO:-LL}"
export CYCLOTRON_RUN_TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
# The 27-DOF URDF is third_party/asimov-1/sim-model/asimov_1_27dof/ when present, else
# asimov-1 at the pinned commit, fetched once into cyclotron's cache (workers share a lock).
# Set ASIMOV_1_27DOF_URDF_PATH to use another file.

RUN_NAME="${RUN_NAME:-lower_body_27dof_payload_4gpu_4096}"
LAUNCH_DIR="${CYCLOTRON_ROOT}/outputs/asimov_lower_body_27dof_payload/${CYCLOTRON_RUN_TIMESTAMP}"
mkdir -p "${LAUNCH_DIR}"
echo "Worker logs: ${LAUNCH_DIR}/workers"
echo "Run: logs/rsl_rl/asimov_velocity_lower_body_27dof_payload_amp/${CYCLOTRON_RUN_TIMESTAMP}_${RUN_NAME}"

"${PYTHON_EXE}" -u -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=4 \
    --log-dir "${LAUNCH_DIR}/workers" --redirects 3 --tee 0:3 \
    scripts/rsl_rl/train.py \
    --task Asimov1-Velocity-LowerBody-27Dof-Payload-AMP-v0 \
    --num_envs 4096 --headless --distributed \
    --max_iterations 10000 --logger tensorboard \
    --run_name "${RUN_NAME}" \
    2>&1 | tee "${LAUNCH_DIR}/train.log"
