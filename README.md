# Physical-AI Workbench

A CPU-tested implementation workbench for the Qwen + pretrained flow-motor research project, with a separate Tau0 startup-demo track.

**Start here:** [v5 research handoff](docs/HANDOFF_V5.md) · [implementation status](docs/IMPLEMENTATION_STATUS.md) · [next-agent instructions](docs/NEXT_AGENT.md)

**Continuation:** [2026-09-29 integration progress](docs/2026-09-29-integration.md).
The four core upstream revisions have now been cloned locally. The native
Qwen3.5-2B processor has been exercised; pretrained policy execution is next.

## What this is—and is not

The local code implements auditable data/action contracts, flow matching, a small reference VLA, a knowledge-insulation gradient boundary with an auxiliary autoregressive objective, staged motor adaptation, optional visual-goal handling, native-adapter scaffolding and numerical qualification tools.

It is **not a trained robot policy**. No pretrained weights, GPU results, closed-loop robot benchmark results or real-world demonstrations are bundled. Native Qwen/W0, A1.5/Tau, Primus/FSDP2/FP8 and real robot integration still require work and qualification. See the status table before scheduling a large run.

Direct cloning was blocked by DNS in the creation environment. Upstream revisions were inspected via the GitHub connector and are recorded in `upstreams.lock.json`; no upstream repository is secretly vendored or claimed cloned.

## Quick start: no models or GPUs needed

In an environment with PyTorch, NumPy, Pillow, PyYAML, safetensors and pytest:

```bash
python -m pip install -e . --no-deps
bash scripts/validate_cpu.sh
```

Pass an output directory (for example `bash scripts/validate_cpu.sh runs/local`)
to preserve the original bundle's reports while recording a fresh validation.

Or create a CPU environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
bash scripts/validate_cpu.sh
```

On the GPU host, **do not replace the ROCm PyTorch build** with a default pip build. Start from the pinned Primus-compatible environment, then install this repo with `--no-deps` and add missing libraries deliberately.

Useful commands:

```bash
pai doctor
pai smoke --output runs/smoke --steps 40
pai make-demo-data --output runs/demo_data
pai audit-data runs/demo_data/manifest.jsonl --output runs/data_audit.json
pai goal-pairs runs/demo_data/manifest.jsonl --output runs/goal_pairs
pai budget --width 256 --height 256 --images 9 --anchor-hz 1
```

The demo data are synthetic colored-image fixtures with manufactured actions, **not a physics simulator**. The smoke model is randomly initialized and tiny; it exists to test mathematics and interfaces, not to replace a pretrained motor prior.

## Train the reference model through the actual data loader

```bash
PYTHONPATH=src python scripts/train_reference.py \
  --manifest artifacts/demo_data/manifest.jsonl \
  --scaler configs/reference_scaler.json \
  --output runs/reference_data --steps 40 --goal-fraction 0.25
```

A two-process CPU/Gloo test is also supported:

```bash
PYTHONPATH=src torchrun --standalone --nproc-per-node=2 scripts/train_reference.py \
  --manifest artifacts/demo_data/manifest.jsonl \
  --scaler configs/reference_scaler.json \
  --output runs/reference_ddp --steps 4
```

This demonstrates reference DDP wiring. It does not demonstrate native VLA training, ROCm/RCCL, FSDP2, tensor/expert parallelism, FP8 or exact distributed resume.

## On a networked host

```bash
python scripts/bootstrap_upstreams.py                     # inspect commands only
python scripts/bootstrap_upstreams.py --execute           # fetch four pinned core repos
python scripts/bootstrap_upstreams.py --names miles miles_diffusion --execute
```

The A-series repository uses `master`; the script uses the observed immutable commit and does not guess the branch. Source code, upstream environments and weights remain separate.

Resolve a model config and immutable revision without downloading its weights:

```bash
python scripts/resolve_hf.py Qwen/Qwen3.5-2B --output weights/qwen2b
```

Add `--download-weights` only when the host has the intended storage/access. Requested newer model identities must be verified on that host; a candidate in `configs/models.yaml` is not a promise that it has been loaded here.

Validate multimodal batches using the actual Qwen processor, without downloading
model weights (requires the `hf` extra):

```bash
python scripts/qualify_qwen_processor.py --output runs/qwen_processor.json
```

`QwenObservationCollator` builds separate observation-only motor inputs and
assistant-supervised auxiliary inputs. It verifies token-prefix agreement after
image expansion and preserves observation tensors across both forwards. Supply
real, versioned action-codec labels upstream; the qualification command uses
synthetic text labels solely to test preprocessing.

## Numerical qualification

```bash
pai make-fixture --output runs/fixture
pai probe runs/fixture --output runs/cpu_a --device cpu --precision fp32
pai probe runs/fixture --output runs/cpu_b --device cpu --precision fp32
pai compare runs/cpu_a runs/cpu_b --output runs/self_parity.json --atol 0 --rtol 0
```

The reference fixture records model/input hashes, forward values, sampler trajectories, gradients, parameter updates and short learning curves. Reports explicitly distinguish CPU self-parity from CUDA/ROCm and native-policy qualification.

Prepare all **native** fixtures and commands before booking NVIDIA time. `configs/nvidia_session.template.json` is not ready to execute until the native jobs and required file hashes are filled in. `scripts/run_nvidia_window.py` enforces a total time budget and stops on failures.

## Key implementation decisions

- **No silent pretrained fallback:** loaders reject unexplained keys, shape mismatches and nonfinite weights. Explicit interface resets are reported.
- **No fake physical alignment:** camera transforms, units, rotations, masks and delta references are explicit. Our 80D schema is not claimed identical to a donor's layout.
- **No privileged evaluation leakage:** real future frames are tagged and rejected in deployment modes; alias episodes share split groups.
- **No “detach only” KI:** the brain receives an auxiliary objective; the motor-to-brain gradient is blocked before trainable bridges. The reference codec is not FAST.
- **No fabricated progress:** proposed subtasks are not recorded as completed until an observed result confirms them.
- **No automatic hardware control:** this repo has no live actuator driver. Simulation and hardware qualification come first.

## Repository layout

```text
src/physical_ai/   runnable reference components and optional native adapters
scripts/           CPU validation, reference training, upstream/HF setup, GPU-session harness
configs/           decisions, candidate models, mixture, experiments and reference fixtures
examples/          worked native integration example
tests/             offline unit/integration tests, not robot benchmark evaluations
docs/              v5 handoff, technical contracts, limitations and next steps
artifacts/         actual CPU reports plus explicitly synthetic data/model fixtures
```

## Measured validation

See `artifacts/VALIDATION_SUMMARY.json`, `artifacts/pytest.txt`, `artifacts/cpu_smoke/report.json`, and `artifacts/ddp_cpu/run.json`. These record what was executed in the creation environment. No throughput extrapolation from CPU fixtures to MI300X is justified.

## Attribution and release

New workbench code is MIT-licensed. Upstream implementations, model weights and datasets remain under their own terms; they are not included or relicensed here. `NOTICE.md` identifies design/code-interface references. Record exact upstream and checkpoint provenance in any derivative release.
