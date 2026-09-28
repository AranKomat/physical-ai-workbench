# First local integration

Repository: https://github.com/AranKomat/physical-ai-workbench

The v5 ZIP was imported as an independent repository. The supplied standalone
handoff is identical to `docs/HANDOFF_V5.md`. The initial import preserves the
creation-environment reports; fresh validation uses `runs/20260929-local`.

## Completed

- All four core source revisions in `upstreams.lock.json` fetched and checked
  out successfully: Primus, InternW0-Delta, InternVLA-A-series and Tau0.
- Original 93 tests reproduced locally. With new collator tests: 98 passed.
- Full CPU validation passed: synthetic training, checkpoint round trip,
  mask preservation, data audit and exact 219-tensor CPU self-comparison.
- Real Qwen3.5-2B processor/tokenizer/template downloaded at revision
  `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- New collator tested with that processor on a batch containing one and two
  images. Prefix lengths were 83 and 151 tokens; assistant-supervised lengths
  were 3 and 5. Changing answers did not change any motor input tensor.
- Native Tau LIBERO and A1.5 Base configuration revisions resolved below.

Local environment: Python 3.12.12, PyTorch 2.14.0, Transformers 5.17.0,
torchvision 0.29.0. CPU execution only. This is not a GPU environment lock.
The first public GitHub CPU workflow also passed.

## Concrete implementation

`src/physical_ai/collation.py` constructs independent native processor batches
for the motor prefix and auxiliary answer. It excludes prefix/padding tokens
from CE, verifies template prefix consistency after image-token expansion,
and checks that both forwards receive identical image tensors and grids.
Tests exercise both padding sides and answer independence.

The caller supplies the auxiliary label text and its provenance. Real
FAST/equivalent action encoding remains open. The processor test's ordinary
text answers are synthetic interface fixtures, not action-token training.
Camera ordering, temporal provenance, state and geometry still need integration
with real dataset windows. This initial collator intentionally has no goal-image
field; goal-conditioning requires a separately audited path.

## First native policy: Tau LIBERO

Use the intact Tau LIBERO export as the first native reproduction. This provides
a bounded simulation target and also advances the separate demo track. It does
not select Tau as the research motor donor; A1.5 versus W0 transfer remains open.

The pinned source `third_party/tau0/configs/libero/README.md` documents:

- Native state: 8D EEF/gripper; native actions: 7D EEF delta/gripper.
- Model-facing state/actions: 40D; active indices 0:9 and 18.
- Rotation conversion: axis-angle to rot6d, then inverse on output.
- Action horizon 10; execute 8 before replanning.
- Two cameras, 180-degree image rotation, 224-square policy inputs.
- Seed 7, eager inference first, separate simulator environment.
- Complete export includes normalization, transforms, prompt and camera labels.

These are source requirements, not results we reproduced. Preserve the entire
native export for the baseline and retain its original schema/normalization.

Verified model metadata:

| Model | Immutable revision | Observed contract |
| --- | --- | --- |
| `sii-research/tau-0-vla-libero` | `ce1dd8f0d917011ce45671fa40edaae1fd6a6ec1` | 40D, horizon 10, BF16, Qwen3.5-2B |
| `InternRobotics/InternVLA-A1.5-base` | `325331b52fc788a7a84419dfcb4930a43b14df1e` | 32D, horizon 50, BF16, Qwen3.5-2B |

Only configuration files were downloaded for these two models. Their different
action widths/horizons confirm that matching Qwen dimensions does not make the
policies interchangeable. W0's inspected source also distinguishes its Base
checkpoint from Wan-only initialization and reinitializes cross-attention K/V
when Qwen conditioning is configured.

## Next substantive steps

1. Stage the complete pinned Tau LIBERO export and its model environment;
   verify one native inference example, then a small seeded LIBERO rollout set.
2. Audit W0/A1.5 checkpoint tensors and source-specific normalization before
   implementing donor transfer. Keep source and weight hashes in each receipt.
3. Complete real action-codec auxiliary targets and connect the collator to
   real observation/action windows. Run one native Qwen/motor BF16 update.
4. Capture native CUDA numerical references during a concentrated GPU window;
   use them to qualify AMD BF16, then FP8.
5. Run short donor-transfer and pretrained-knowledge preservation comparisons.

The old EmbodiedSWE GPU rental was being stopped; this work did not restart it
or assume access to another GPU. No pretrained weights, robot rollouts, AMD
results, FP8 results or physical capability are claimed by this continuation.

Selected receipts are under `artifacts/local_20260929/`; complete generated
fixtures remain in ignored `runs/`. Reproduce with:

```bash
bash scripts/validate_cpu.sh runs/local
python scripts/qualify_qwen_processor.py \
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --output runs/qwen_processor.json
```
