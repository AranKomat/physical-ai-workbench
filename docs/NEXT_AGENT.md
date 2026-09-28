# Next research agent: start here

## Latest continuation: 2026-09-29

Read `2026-09-29-cpu-completion.md` and `CPU_WORK_PLAN.md` first. These supersede
the older preparation gaps below: real LIBERO captures, 50 real FAST windows,
vocabulary resume, full donor inventories and reduced native CPU adapters now
exist. W0 Base uses `mot` / `mixtures.action.` with source context width 2048;
use `configs/w0_base_audited.json` and preserve its trained K/V projections.
Pretrained inference and research training remain unexecuted. Do not rent or
contact a GPU host until the user supplies one. Follow `GPU_RUNTIME_PREPARATION.md`
and `EVALUATION_PROTOCOL.md` for the next bounded hardware session.

## 1. Confirm the handoff, not the old chat claims

Read `HANDOFF_V5.md` and `IMPLEMENTATION_STATUS.md`. The prior reported saved implementation was missing in this runtime; this is a fresh rebuilt deliverable. The v5 plan supersedes the obsolete v1 random-head/4B-first plan.

Run:

```bash
python -m pip install -e . --no-deps
bash scripts/validate_cpu.sh
```

These are reference/contract checks. Do not advertise their loss curves as physical-AI capability.

## 2. Fetch source, preserving the GPU environment

```bash
python scripts/bootstrap_upstreams.py --execute
```

Do not install all donor repos into one environment. Keep Tau proposal, Tau world model, native Tau low-level, native W0 and the AMD training environment separate until incompatibilities are understood. The Tau world-model guide explicitly uses a separate environment. Avoid pip commands that replace ROCm torch.

Pin the image digest, ROCm/driver, PyTorch, Transformers, attention kernels and every model revision in the run manifest. The code lock is not a complete GPU dependency lock.

## 3. Reproduce one native policy before surgery

Choose A1.5, W0 or Tau in its original interface. Obtain weights and data/proprocessor artifacts. Run its official example or simulator recipe first.

For W0:

```bash
pai inspect-checkpoint /path/to/pretrain.pt --key-path <actual_tensor_container> \
  --prefix <actual_action_expert_prefix> --output runs/w0_keys.json
```

The placeholders are intentionally not guessed. Inspect the actual checkpoint and source save/load code. `torch.load(weights_only=True)` may reject a legacy container; do not casually disable it. Convert a trusted checkpoint in an isolated environment to a tensor-only state dict if necessary.

Verify the original action dimension and normalization, flow timestep scale, sign, shifted schedule and weighting. `configs/w0_constructor.json` records source architecture values but its dynamic action dimension still needs checkpoint verification.

## 4. Complete the smallest real Qwen/native-motor training step

`examples/native_composition.py` shows the implemented loader/composition boundary. The remaining primary work is a production multimodal collator, real discrete auxiliary labels and audited native checkpoint extraction.

Use separate Qwen prefix and auxiliary CE inputs. Do not expose target action tokens or privileged future labels through a shared teacher-forced prefix. Preserve real image-token masks. Do not assume text-token counts equal the processor's effective multimodal sequence length.

First run in BF16 with eager/known-good kernels and no FP8. Use a tiny number of complete, verified action windows. Do not start a giant dataset download or 27B run here.

## 5. Prepare a concentrated NVIDIA window

Extend `configs/nvidia_session.template.json` with all native jobs and required file hashes. The template is deliberately `ready_for_cuda: false`. The reference toy probe alone is not enough.

Before accessing CUDA hardware, make checkpoints, fixtures, images and code local; resolve environment packages; test command imports and memory requirements. Capture forward, velocity, sampler trace, selected gradients, one-step updates and a short matched curve in one window. Save the outputs for later AMD regression checks.

A smaller CUDA GPU may suffice for the component being tested, but check actual model/context memory first. Full W0 and its isolated ActionDiT have different memory requirements. Do not schedule H200/B200 dependence unnecessarily.

## 6. Only then enable the real AMD optimization path

Portable PyTorch running inside a Primus container is the first milestone. A native Primus/Megatron/FSDP2 backend is a separate milestone.

Test distributed gradients, unused-parameter behavior, checkpoint resume, image preprocessing and action masks. Compare actual VLA BF16 and FP8 short learning and closed-loop metrics. Do not certify FP8 from a reference-model loss or an upstream FLUX example.

## 7. Keep experiments small and causal

First resolve donor transfer (A1.5 versus W0; Tau only if useful). Then use protected/insulated learning with a brief unprotected control. Adopt the already-selected physical alignment and metadata defaults rather than repeat every paper ablation.

Use public semantic/OOD benchmarks and retained grounding capability, not easy IID success alone. Define fair token and node-hour budgets before backbone comparison. Different donors are different architectures; disclose that confound.

## 8. Keep the Tau demo independent

Run Tau's native proposal and native low-level policy without forcing our 80D schema into them. The proposal client and world command builders are ready to connect to native processes.

The world model may help select a subtask; it is not automatically an accepted low-level goal-image modality. For our new goal-conditioned policy, compare no goal, real-future oracle, goal-free inference after privileged training, and generated-goal inference separately.

Do not send physical commands until a separate hardware bridge enforces limits, watchdog, stop conditions and appropriate supervised testing.

## 9. Report honestly

Each run should record source/weight/config hashes, observation privileges, data groups, actual token counts, consumed windows, precision/optimizer state, throughput, losses, closed-loop success and failure examples.

No statement such as “trained on AMD,” “preserves W0's physical prior,” “outperforms Tau,” or “four-step goals are good enough” is supported until the corresponding run has actually been performed.
