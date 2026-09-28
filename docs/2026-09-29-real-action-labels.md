# Real action labels and native fixture preparation

## What ran

Three real A2D windows from Tau's bundled gong dataset passed through native
video decoding and native action conversion, the public FAST codec, and our
Qwen observation/auxiliary collator. No pretrained forward or gradient update
was executed. This is the first real action-label integration, not a trained
policy or an A1.5 reproduction.

Pins:

- Tau source: `f1665fbaf624d1468b168e3a54cccc9e326212d9`.
- FAST: `physical-intelligence/fast` at
  `ec4d7aa71691cac0b8bed6942be45684db2110f4`.
- Qwen processor: `Qwen/Qwen3.5-2B` at
  `15852e8c16360a2fea060d615a32b45270f8a8fc`.

FAST's remote implementation was read before execution. It uses DCT,
quantization and BPE; the probe enforces the reviewed source hash. Input actions
are the native route's `extras.action.q_norm`, using its supplied normalization
statistics, not invented scaling or fitted test-set statistics. The probe
fails when quantile statistics are absent. It preserves the native 30x40
interface and masks; it does not call these A1.5's 50x32 actions or research 80D.

| Filtered anchor | FAST codes | Prefix tokens | Supervised tokens | All-channel normalized RMSE |
| --- | ---: | ---: | ---: | ---: |
| 0 | 19 | 255 | 21 | 0.00327 |
| 2742 | 40 | 255 | 42 | 0.00753 |
| 5483 | 36 | 255 | 38 | 0.00655 |

Supervision includes two template-ending tokens in addition to action codes.
These RMSE values include zero padding. The receipt also reports active-joint
RMSE, per-dimension RMSE, masks, maximum error and normalization hash. Do not
interpret quantization error as a physical success rate or closed-loop bound.

All action codes survived template processing exactly. Replacing the auxiliary
answer with a different action token left every motor-prefix input unchanged.

## Important findings

FAST's native decoder catches errors and can return zero actions. The wrapper
rejects coefficient clipping, invalid codes, lossy BPE round trips, nonfinite
outputs, and decoded actions inconsistent with the checked DCT coefficients.
There is no empty-token or silent-zero fallback and no action-token truncation.

A1.5's source supplies `tokenizer_file` explicitly when loading FAST under
Transformers 5; our loader follows that compatibility pattern.

Tau's exported tokenizer does not share A1.5's robot-action vocabulary. The
research probe instead adds 2,048 robot-action tokens to the base-Qwen tokenizer
and verifies IDs 248077 through 250124, following the inspected A1.5 mapping.
Tau's demo tokenizer/checkpoint is not modified. No model embeddings have yet
been resized or trained: the first real model step must do that explicitly and
verify input/output embedding dimensions and initialization.

Reproduce in the isolated CPU environment from `2026-09-29-native-cpu.md`, after
staging the pinned FAST files and Qwen processor:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv-tau/bin/python scripts/qualify_fast_labels.py \
  --upstream third_party/tau0 --fast weights/fast \
  --output runs/fast-real-labels.json
```

Receipt: `artifacts/local_20260929/fast-real-labels.json`.

## LIBERO fixture path

`scripts/prepare_libero_fixture.py` accepts a raw observation NPZ containing
agentview_image, robot0_eye_in_hand_image, robot0_eef_pos, robot0_eef_quat (xyzw),
and robot0_gripper_qpos. It uses the pinned native client's actual image and
quaternion functions, then writes observation.npz plus provenance.json.

Required metadata: input kind (simulator or synthetic), task, suite, episode and
seed. It rejects nonfinite/misshaped proprioception, unnormalized quaternions,
dirty or wrong upstream revisions, and output overwrites. Input-kind metadata
is a caller declaration, not proof of simulator provenance.

The converter was exercised on a synthetic patterned-image fixture and identity
rotation. Output images matched native 180-degree rotation and bilinear resize;
the 8D state matched expected values. **No real LIBERO capture exists yet.**
Synthetic success is only a format test. A genuine capture must retain simulator
revision, task ID, initial-state identity and observation timing in addition to
the converter's manifest before claiming matched native reproduction.

## Still missing

- Genuine LIBERO capture, pretrained inference and closed-loop reproduction.
- Audited research state/geometry inputs, temporal masks and scalable batching.
- Research action-codec normalization across donors/embodiments and held-out data.
- Qwen embedding expansion and a real insulated Qwen/motor BF16 training step.
- CUDA/ROCm parity and all physical-capability comparisons.
