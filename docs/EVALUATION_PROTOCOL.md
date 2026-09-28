# Frozen first-session evaluation protocol

Version: 2026-09-29. Preparation is GPU-free; execution is not yet reported.
Changing a condition requires a new run ID and a recorded reason, not overwriting
failed runs. Do not pool demo and research results.

## D0: intact native Tau reproduction

- Source/checkpoint pins and observation conventions are those audited in the
  native Tau receipts. No research action schema, adapter or GPT supervision.
- LIBERO spatial task IDs 0, 1, 2; initial-state indices 0, 1, 2; seed 7.
  Nine episodes, all attempted. This is a smoke subset, not the full benchmark.
- Native 256-square renders, 180-degree rotation, 224-square model images,
  native state and normalization, horizon 10, replan after 8 actions, 10 initial
  settling actions. Use the upstream spatial-suite horizon, not an invented cap.
- Eager inference first. Warmup and cold model loading measured separately.
- Before rollouts: one finite native 10x7 action chunk on a captured observation,
  then repeat with identical seed/input and retain differences rather than
  assuming determinism. No robot hardware commands are involved.
- Log independent simulator success, error, timeout and abort separately.
  Report successes / all nine attempted episodes and the number of errors.
  Retain video and raw commands for failures as well as successes.
- Record p50/p95 inference latency, peak allocated/reserved VRAM, total runtime,
  source/package/model hashes and renderer. Small sample latency is descriptive.
- Stop on nonfinite/wrong-shaped output or repeated infrastructure errors;
  do not count an unattempted episode as a completed failure or retry silently.

## R0: research plumbing qualification

The 50 A2D windows across 25 episodes are integration fixtures only. Supplied
upstream normalization may have seen all those episodes. No held-out performance
or generalization claim is permitted from this subset.

1. Load one complete-window batch and verify its provenance/hash, masks, camera
   order, action vocabulary and no teacher-forced targets in motor inputs.
2. Expand Qwen vocabulary before optimizer creation. Audit all loaded/reset
   parameters; reject unexplained missing keys and silent initialization.
3. Fixed seed 7; fixed saved noise/timesteps; FP32 CPU small-model checks first,
   then BF16 eager GPU. Capture prefix outputs, native velocity, sampler trace,
   selected gradients and one optimizer update.
4. Insulated flow must not give Qwen gradients, but CE must. Bridge and motor
   must receive flow gradients. Run a matched uninsulated control solely to
   confirm the boundary. Do not infer retained knowledge from this test.
5. A maximum of 100 training steps on at most 256 preselected complete windows
   for the first pipeline run. No model-ranking conclusion from training loss.

## R1: donor-transfer comparison, conditional on R0

A1.5 and W0 have different architectures and native representations. This is
a transfer-recipe comparison, not a pure initialization ablation. Before any
run, freeze the target representation and explicitly audit each conversion.
Preserve source normalization for native baselines. Never infer compatibility
from action dimension alone.

Use the same Qwen version, split groups, consumed complete windows, target task
suite and step budget. Report both consumed tokens and GPU hours. Record frozen,
reset and trainable tensors, and distinguish intact-native from transferred
results. A1.5's per-layer prefix cache is not interchangeable with W0's single
context tensor; adapting that difference must be documented and measured.

Use disjoint source episodes/scenes for train, validation and test; compute
normalization only on train or disclose inherited statistics. Select checkpoints
on validation, never test. A small integration fixture is insufficient to freeze
a scientifically meaningful held-out suite, so R1 remains gated on data selection.

## Numerical and reporting rules

- CUDA references precede ROCm comparison. CPU self-parity is not cross-device
  evidence. Save precision, backend and optimizer state alongside every tensor.
- Initial diagnostic tolerances: FP32 atol=1e-5/rtol=1e-4, BF16
  atol=2e-2/rtol=5e-2. These are screening thresholds, not acceptance criteria for
  physical control. Report absolute/relative errors and task behavior separately.
- No FP8 before BF16 baseline and short training stability pass.
- For paired policies, retain per-start outcomes, not just aggregate success.
  With nine episodes, avoid confident ranking or extrapolation to all tasks.
- A negative result remains a result. Do not replace a failed condition with a
  new prompt, easier start, altered controller or longer horizon under the same ID.
