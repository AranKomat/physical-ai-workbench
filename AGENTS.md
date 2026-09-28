# Working in Physical-AI Workbench

Read `docs/NEXT_AGENT.md`, `docs/IMPLEMENTATION_STATUS.md`, and `docs/HANDOFF_V5.md` before changing model lineage or scheduling GPU work.

This is a **CPU-tested workbench**, not a pretrained robot model. The immediate milestone is a native donor reproduction followed by the smallest audited Qwen/native-motor BF16 training step. Keep the separate intact-Tau demo path separate from the research policy.

## Development

```bash
PYTHONPATH=src python -m pytest tests -q
bash scripts/validate_cpu.sh
```

Never replace a working ROCm PyTorch install with the ordinary pip dependency resolution. Use the pinned external environment, install this package with `--no-deps`, and resolve optional dependencies deliberately.

## Boundaries to preserve

- Do not silently accept checkpoint mismatches, invent FK/calibration, or call our slot ordering a donor's schema.
- Keep motor prefix inputs separate from teacher-forced action answers. Keep privileged future goals out of deployable evaluation.
- KI needs an auxiliary backbone objective; detaching flow alone is not its full recipe.
- Require observed completion before advancing execution memory.
- Prepare all native fixtures before concentrated NVIDIA access. CPU self-parity cannot stand in for CUDA/ROCm parity.
- The native source pins are in `upstreams.lock.json`. Weight revisions, full dependency locks and GPU results remain to be collected.
- No real actuator command should be added without an independent hardware safety/limit/watchdog layer.
- Preserve upstream attribution; do not hide model lineage or conflate inherited results with this project's contribution.

Log unresolved assumptions and measured results in the run report. Update implementation status when native integration, hardware qualification, or benchmarks actually pass.
