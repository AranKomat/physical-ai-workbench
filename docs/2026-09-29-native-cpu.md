# Native CPU qualification

This continues `2026-09-29-cpu-preparation.md`. No GPU, pretrained model forward,
training step or simulator rollout was run.

## Dependency workaround and its scope

The unmodified pinned Tau policy, ModelBuilder and Tau0VLAModel now import.
LeRobot 0.4.1 was installed with `--no-deps`, retaining Hub 1.33.0 for
Transformers 5.5.4. We installed the actual data-path dependencies separately.
This deliberately violates LeRobot's declared Hub upper bound. Tests below
qualify a narrow offline surface, not arbitrary LeRobot functionality.

A second upstream conflict exists: datasets 4.1.1 requires PyArrow >=21, while
Tau's requirements pin 20. We retained PyArrow 25.0.1. Native PyAV decoding
works on this Mac; TorchCodec/CUDA decoding has not been qualified. LeRobot's
unrelated training, visualization and hardware extras were not installed.

Recreate in a dedicated environment, never over a working ROCm installation:

```bash
uv venv --python 3.12 .venv-tau
uv pip install --python .venv-tau/bin/python -r configs/tau-cpu-requirements.txt
uv pip install --python .venv-tau/bin/python --no-deps lerobot==0.4.1
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv-tau/bin/python scripts/qualify_tau_cpu.py \
  --upstream third_party/tau0 --output runs/tau0-native-cpu.json
```

The requirements file pins direct dependencies; the execution receipt records
all installed package versions. It is not a cross-platform wheel/hash lock.

## Real data result

The unmodified `agibot_world_gong_ft` route reads its bundled A2D robot records
and three actual camera videos through the native FinchDataLoader and LeRobot.
Its own instruction-span filtering retains 5,484 of 9,469 anchors. This replaces
the raw unfiltered window count as the useful native-loader count; it does not
mean every anchor contains 30 distinct future observations.

At filtered indices 0, 2742 and 5483:

- Three 224x224 RGB uint8 views decode: head, wrist_left and wrist_right.
- Native state is 40D; native action horizon is 30x40.
- State and actions are finite; all inactive slots are zero.
- Sixteen slots are active: 18, 19, 24-30 and 32-38.
- The prompt comes from upstream's annotated task spans. It declares G1 joint
  control because that is the upstream A2D-to-G1 adapter recipe, not a mapping
  we invented or a claim that the recorded robot is physically G1.

Seed 7 and native training augmentation are retained. Image/action digests and
full environment versions are in `artifacts/local_20260929/tau0-native-cpu.json`.
Only three windows were decoded by this qualification, not all 5,484 anchors.
This is real-data conversion evidence; it does not qualify the LIBERO checkpoint
on A2D data or convert these actions into our research 80D schema.

## Next meaningful work

Regression evidence: all 101 workbench tests pass, plus 23 upstream tests
(and two subtests) across LIBERO adapter, deploy wire and LIBERO evaluation
accounting. The latter use mock simulators for accounting tests: they are not
LIBERO rollouts. Two FastAPI/Pydantic deprecation warnings occur. Both combined
runs also printed a multiprocessing resource-tracker KeyError at interpreter
shutdown after reporting success and exit code 0; cleanup on macOS remains a
qualification caveat, not silently classified as a perfectly clean run.

1. Capture a matched LIBERO observation fixture and qualify checkpoint loading
   on NVIDIA. Preserve native normalization and image processing.
2. Connect real source-native windows to the Qwen collator with an audited
   action codec. Current ordinary text auxiliary fixtures are not action labels.
3. Inspect A1.5/W0 pretrained motor keys and transfer contracts independently
   of the intact Tau demo path.
4. After a native baseline, capture CUDA numerical references for AMD checks.

Full-model inference remains deferred rather than forcing a 3B policy into this
16 GB Mac. Do not treat successful imports or adapter tests as model results.
