#!/usr/bin/env bash
# One-shot upload of the 2026-10-26 lower-body 27-DOF run to the Hugging Face Hub, with its model card.
# Run with user/ariel/lowerbody-27dof checked out: that branch registers the run's task.
# Usage: scripts/share_lower_body_27dof.sh [repo_id]   (needs a prior `hf auth login`)
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN=$ROOT/logs/rsl_rl/asimov_velocity_lower_body_27dof_amp/2026-10-26_18-59-36_lower_body_27dof_4gpu_4096
CARD=$ROOT/docs/model_cards/asimov1-locomotion-26-10-26-lower-body-27dof/README.md
REPO=${1:-Menlo/asimov1-locomotion-26-10-26-lower-body-27dof}
TITLE="Asimov 1 locomotion policy checkpoint — 2026-10-26 lower-body 27-DOF, torso IMU"

cd "$ROOT"
./cyclotron.sh --share "$RUN" --repo-id "$REPO" --checkpoint model_9999.pt \
    --task Asimov1-Velocity-LowerBody-27Dof-AMP-v0 --title "$TITLE"
hf upload "$REPO" "$CARD" README.md --commit-message "Add model card"
echo "Done: https://huggingface.co/$REPO"
