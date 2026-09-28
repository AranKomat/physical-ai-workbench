# Executed validation — 2026-09-29

All results below were produced locally, not copied from an upstream paper.

| Check | Result | Scope |
|---|---|---|
| Automated tests | **93 passed**, zero failures/skips | Data, geometry, masks, gradients, checkpoints, goals, native-interface fakes, session preflight |
| Reference fixed-batch training | 40 CPU FP32 optimizer steps; flow loss 1.023472 → 0.080292 | Tiny random synthetic reference model; not a robot benchmark |
| Checkpoint round-trip | Maximum output difference 0.0 | Same CPU checkpoint/noise/input |
| Inactive channels | Maximum absolute sample value 0.0 | Mask invariant |
| Episode Dataset → training | 8 CPU FP32 steps, with optional real-future fixture inputs | Synthetic episodes through the implemented loader |
| BF16 autocast | 8 CPU steps completed | CPU only, not AMD or FP8 qualification |
| Distributed reference training | 2 processes, Gloo, 4 steps | CPU DDP, not GPU sharding or exact resume |
| Golden capture self-comparison | **219 tensors**, exact equality at atol=rtol=0 | Same CPU implementation twice, not an independent oracle |
| Upstream/bootstrap commands | Six locked repositories, dry runs passed | No repositories downloaded |
| NVIDIA batch plan | Dry run and preflight tests passed; template not marked ready | No CUDA jobs executed |
| Editable package install | Passed with no dependency replacement | Creation environment |

Environment: Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5. The observed package list is saved separately; it is not a recommended GPU lock file.

## Evidence files

- `artifacts/VALIDATION_SUMMARY.json`
- `artifacts/pytest.xml` and `artifacts/pytest.txt`
- `artifacts/cpu_smoke/report.json` and its learning curve
- `artifacts/data_training/run.json`
- `artifacts/data_training_bf16/run.json`
- `artifacts/ddp_cpu/run.json` and `artifacts/ddp_cpu.log`
- `artifacts/cpu_self_parity.json`
- `artifacts/nvidia_dry_run/session.json`

## What has not been demonstrated

No native pretrained checkpoint was loaded. No GPU, simulator, real robot, image generator, robotics LoRA or reinforcement-learning experiment ran. The native wrappers were tested against local interface fakes, not real model implementations. Real FAST targets, the production collator, donor adapters, streaming data conversion and Primus integration still need coding as detailed in the handoff.

CPU loss reduction establishes learnability of the reference objective only. Exact self-parity establishes deterministic capture/comparison mechanics only. Neither establishes transfer, OOD success, numerical equivalence to NVIDIA, or retained pretrained capability.
