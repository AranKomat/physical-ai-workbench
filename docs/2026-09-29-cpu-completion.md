# Eight-item GPU-free preparation report

No rented GPU was used. Native-policy inference, pretrained training and
CUDA/ROCm qualification remain unexecuted. Local simulator rendering used the
Mac's OpenGL graphics, not a software-only renderer.

## 1. Genuine LIBERO observations

Three real LIBERO spatial-task captures: task IDs 0, 1, 2, initial-state index 0,
seed 7, ten native settling actions. Source pin:
`8f1084e3132a39270c3a13ebe37270a43ece2a01`. Tau's pinned native client supplies
image processing and 8D EEF/gripper conversion. No policy ran.

Local artifacts: `runs/libero-captures-v4/task-*/`. Each has raw NPZ, processed
NPZ, two PNGs and a provenance manifest with BDDL, initial-state and observation
hashes, simulation time and package versions. All six images pass nonblank
checks; task 0's two images were visually inspected. Simulator time is 0.5 s.
Initial states were loaded with restricted PyTorch unpickling and explicit NumPy
allowlists, not an unrestricted pickle fallback.

Environment: Python 3.10, torch 2.6.0, robosuite 1.4.0, MuJoCo 3.2.3,
NumPy 1.24.4. `configs/libero-sim-requirements.txt` includes extra import-time
dependencies absent from the minimal upstream example. Private-macro, Gym-age
and missing-demonstration-dataset warnings remain; captures do not need the
demonstration download. Earlier failed import attempts are retained separately.

## 2. Real donor checkpoints

| Donor | Immutable revision | Bytes | Tensors |
| --- | --- | ---: | ---: |
| A1.5 Base | 325331b52fc788a7a84419dfcb4930a43b14df1e | 5,385,900,721 | 950 |
| W0 Base | ca534d4f3c9c7131205b4ee37e965b3ef21ffb5f | 12,423,420,182 | 1,660 |

Both full files match their published Hugging Face LFS SHA-256 hashes. No
unrestricted pickle loading was used. A1.5 uses safetensors header inspection;
W0 uses restricted memory-mapped loading onto the meta device. Header inventory
does not establish finite tensor values or working inference.

A1.5 has 319 action-expert tensors; its action projection is 1024x32 and horizon
50. Its checkpoint includes learned video-branch interfaces. State is tokenized
in the released config, so there is no standalone state projection to extract.
Source normalization/field transforms remain part of its native contract;
the top-level IDENTITY mapping does not mean raw arbitrary robot actions work.

W0's actual container is `mot`, prefix `mixtures.action.`. All 824 action tensor
keys/shapes match the native architecture reconstructed on the meta device:
1,209,779,280 parameters, 80 action dimensions, 30 layers, hidden width 1024,
24 attention heads of width 128. Its 2048-wide K/V projections are already
trained for raw VLM context. The residual text-embedding module still has
4096 input width; it is not the operative raw-VLM context width.

These findings corrected the loader: configure source VLM conditioning before
strict loading and do not overwrite pretrained K/V afterward. Different target
context width is rejected until an explicit projection-transfer recipe exists.
Equal 80D width does not certify compatibility with our research action schema.

## 3-4. Real batching and FAST coverage

`runs/real-batches-v1/` contains 50 complete native windows across all 25 bundled
A2D episodes, saved as 13 tensor-only reloadable batches and an expanded Qwen
processor. The wrapper selects anchors whose full 30-frame targets remain
inside source ranges and episodes. It records source identity, normalization,
camera order, requested camera timestamps, action timestamps and boolean masks.

FAST lengths span 19-73 codes. Mean active-dimension normalized RMSE is
0.0119734, maximum per-window active RMSE 0.0207724. All labels survive Qwen
collation without truncation. The earlier three-window test separately verifies
that changing labels leaves every motor-prefix tensor unchanged. Decoder/BPE
corruption, clipping and silent-zero fallbacks fail explicitly.

Native camera decoding uses LeRobot's requested timestamp/tolerance semantics;
we do not claim an independent sensor synchronization calibration. Native
training image augmentation remains enabled with seed 7. These batches are
integration fixtures, not held-out evaluation data or a conversion to 80D.
Inherited normalization may include all source episodes.

## 5. Vocabulary expansion

Added explicit append-only expansion, mean initialization over old tokenizer
rows, input/output weight-tying checks, and idempotent resume. Tests cover tied
and untied heads and existing padded embedding capacity. A tiny actual
Qwen3_5ForConditionalGeneration instance also passed expansion, nonzero new-row
gradients, an optimizer update, exact saved-weight reload and optimizer reload.
No pretrained Qwen weights were changed. Optimizers must be built after resize.

## 6. Donor interfaces

W0: native reduced random ActionDiT forward exactly matches the wrapper, 66
gradient tensors are finite, state conditioning receives gradient, and a
preconditioned native checkpoint round trip preserves every value. The full
real Base architecture matches the audited tensor inventory. Actual pretrained
forward/backward remains pending.

A1.5: added a separate native-cache adapter. It rejects wrong shapes,
nonfinite values, missing caches and action-answer tokens in the prefix. It is
not interchangeable with W0's final-context interface. The actual native code
passed a reduced two-layer mixed linear/full-attention prefix-cache
forward/backward with synthetic text and one learnable token.

Important runtime distinction: A1.5 requires its own patched Transformers
5.2.0 module and torch 2.10.0, unlike Tau's Transformers 5.5.4 / torch 2.7.1.
The successful local environment used Python 3.12 rather than upstream's 3.11;
that deviation is not a claim to reproduce their full CUDA environment.
The source-pinned replacement Qwen module was copied only into `.venv-a15`.
The receipt records its hash. No CUDA attention kernels were installed; the
native PyTorch fallback ran. Earlier tiny-test failures with no linear layer
or zero learnable tokens were invalid reduced configurations and were corrected
without modifying upstream source.

## 7. Evaluation plan

`EVALUATION_PROTOCOL.md` fixes native Tau's initial nine-episode smoke subset,
separates integration from held-out research, defines token/step budgets and
records errors separately from physical failure. Donor comparisons must disclose
architecture differences. No success rate is reported yet.

## 8. GPU bundle

`prepare_gpu_bundle.py` assembles explicit file hashes and a transfer manifest,
excluding caches and Mac virtualenvs. It creates three bounded native Tau
observation jobs and one W0 numerical job. `probe_w0_numerics.py` captures
velocity, selected gradients, an SGD update and a sampler trace using fixed
synthetic inputs, not physical trajectories. It preserves the audited source
context projections and adds an explicitly new state bridge.

`ready_for_cuda` intentionally stays false until the future host passes driver,
runtime import and memory preflight. The prepared commands have not run on a
GPU. A1.5 full inference and research Qwen/motor training remain separate gates,
not silently represented by the W0 motor probe. Runtime instructions are in
`GPU_RUNTIME_PREPARATION.md`.

## What remains after preparation

Native Tau inference and seeded rollouts; actual pretrained donor transfers;
physically audited research action conversion; Qwen/native-motor BF16 training;
held-out data selection; CUDA references; ROCm qualification; and any claims
about physical behavior. The CPU checklist is preparation, not those results.
