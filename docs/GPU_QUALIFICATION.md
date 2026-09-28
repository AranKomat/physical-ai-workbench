# Bounded GPU qualification

Goal: rule out silent implementation/backend/precision bugs, not reproduce every upstream paper.

## Layers of evidence

1. CPU unit tests: contracts and math only.
2. Native upstream checkpoint in original environment: establishes the intended reference.
3. Same checkpoint and processed tensors, CUDA BF16 versus ROCm BF16: port parity.
4. Same real VLA, ROCm BF16 versus actual FP8 path: precision qualification.
5. Small matched closed-loop task set: detects amplification of small open-loop differences.

Do not collapse these into a single “passed” flag. `validation.py` reports scope and hardware explicitly.

## Fixed fixture contents

Store weights/hash, config, processor revision, input IDs, image tensors/grids, calibration, state, action targets/masks, model-space scaling, fixed Gaussian noise, flow timesteps, and optional context states captured at the native motor boundary. Preserve RNG states and disable stochastic inference differences where possible.

W0 extraction adds another test: original MoT coupling versus standalone reconditioned expert is a *model-change* comparison, not hardware parity.

## Metrics

Compare full velocity fields, a few intermediate activations, per-step action denoising, masked loss, selected parameter gradients and update deltas. Include max/p99 absolute error, relative L2 and gradient cosine. Report normalized-space and physical-unit action errors.

Calibrate tolerances using same-backend repeated runs and numerical expectations. CPU exact self-parity is useful for regression, but its zero tolerance is not appropriate as a general BF16 cross-hardware criterion. Small absolute errors near zero need separate treatment; relative errors alone can mislead.

## NVIDIA access pattern

Use a single prepared session with a total budget, e.g. 7200 seconds. No provisioning or background task is implied by this configuration. The runner is a script for the user/research agent's GPU host.

```bash
python scripts/run_nvidia_window.py --plan configs/nvidia_session.template.json
# After actual native fixtures/jobs are prepared and ready_for_cuda is set:
python scripts/run_nvidia_window.py --plan runs/ready_cuda_plan.json --execute \
  --budget-seconds 7200 --output runs/cuda_reference_session
```

The runner stops after a failure and saves logs. Bring everything local first; model downloads should not consume the concentrated session. Retain CPU reference fixtures separately from native captures so their provenance cannot be confused.

## FP8 and RL

Do not begin by simultaneously changing model architecture, checkpoint layout, distributed backend and precision. BF16 first, then one change at a time. FP8 compute is not equivalent to FP8 optimizer state. This bundle does not implement a validated FP8 path.

RL begins in BF16 after strong SFT. It adds policy-distribution/likelihood and rollout-consistency questions that are independent of supervised flow training. A deterministic ODE's MSE is not its policy log probability.
