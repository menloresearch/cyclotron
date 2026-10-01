# Community policies: get-up and safe-fall (OpenHorizon Labs)

Two Asimov 1 policies from OpenHorizon Labs, shared on the Hugging Face Hub:
https://huggingface.co/openhorizon-labs/asimov1-getup-safefall

- **get-up**: from any fallen posture back to this repo's walking-policy
  standing pose, ready to hand over to it.
- **safe-fall**: once a fall is detected, shapes it so the head, torso, pelvis
  and knees take as little impact as possible.

They use the same action convention as the walking policy (23 joint position
offsets, scale 0.25, 50 Hz) and the same PD gains and armature, but a
different observation (5-step history, 375 inputs), so `cyclotron.sh --view`
cannot load them. The Hub repo includes a standalone MuJoCo runner (numpy,
mujoco, onnxruntime) that runs them on `asimov-1/sim-model`.

Trained in mjlab (MuJoCo Warp). Source: https://github.com/openhorizon-labs/asimov-v1-policies

## Results (simulation)

Full domain randomization, flat and bumpy ground, 2048 episodes per test:

| Metric | Result |
|---|---|
| Get-up success | 99.8% |
| Get-up median time to stand | 1.70 s |
| Safe-fall impact on head, torso, pelvis, knees vs a passive fall | 94.7% lower |
| Get-up success after a safe-fall | 99.7% |
| Second engine (CPU MuJoCo): get-up success / safe-fall impact reduction | 100% / 92.3% |

Neither policy has run on a robot. Both need joint commands to be accepted
during and after a fall, which the first firmware release does not allow.
Discussion: https://forum.menlo.ai/t/questions-on-running-custom-get-up-safe-fall-policies-on-asimov-1/69
