# Implementation status

This file is the boundary between executable work and planned integration. A CPU pass is not a pretrained-policy result.

Continuation on 2026-09-29: four core pinned sources cloned, 108 local tests pass,
and the real Qwen3.5-2B processor passes multimodal prefix/CE separation checks.
See `2026-09-29-integration.md` for evidence and the first native reproduction plan.
See `2026-09-29-cpu-preparation.md` for the completed Tau weight audit, real A2D
record audit, deferred GPU access and unresolved native dependency conflict.
The follow-up `2026-09-29-native-cpu.md` records a tested, explicitly overridden
CPU dependency setup: native model imports, 23 upstream tests, and three decoded
real A2D windows. GPU inference is still unexecuted.
`2026-09-29-real-action-labels.md` adds real FAST/Qwen label collation and a
synthetic-tested native LIBERO fixture converter; genuine capture remains open.

| Area | Implemented | Executed here | Remaining |
|---|---|---|---|
| Physical action contract | Versioned 80D research schema, masks, explicit native channel mapping | CPU tests | Audit each donor/source's actual slot semantics |
| Geometry | SO(3)/SE(3), camera EEF delta conversion and inverse | CPU round-trip/edge tests | Source FK, wrist-camera timestamp conventions, native controller audit |
| Data | JSON/NPZ episodes, validation, split groups, sparse history, windows, family sampler; native Tau data probe | Synthetic episode training; three real A2D windows through native LeRobot/PyAV, camera decoding and 40D conversion | Connect native windows to research collator and action codec; temporal/geometry audits |
| Quality | Numeric checks, unknown metadata, frozen/static diagnostics | CPU tests | Calibrated semantic/kinematic quality judgments |
| Reference policy | Tiny image/text brain + cross-attention flow motor | Synthetic CPU learning | Not a pretrained VLA |
| KI mechanics | Stop flow gradients before bridge; auxiliary AR loss; checked FAST labels | Gradient/leakage tests; three real native windows encoded with FAST and Qwen CE masks | Real pretrained CE/flow forward-backward, embedding expansion and donor integration |
| Motor preservation | Reference parameter groups/stages, LoRA residual | CPU tests | Native donor-specific grouping and retention evaluation |
| Layer routing | Learned scalar layer mixer | CPU tests | No LayerRoute reproduction or claimed gains |
| Native W0 | Signature-matched wrapper; strict extraction/loading; explicit time/sign conversion | Fake-native contract tests only | Download actual Base weights, inspect keys and normalize; import and qualify native source |
| Native Qwen | Lazy HF wrapper, prefix/CE separation, composition with motor | Fake-HF interface tests only | Install HF, load actual weights, validate multimodal processor and memory |
| Native multimodal collation | Observation processing, assistant-only CE masks, independent prefix/auxiliary forwards | Pinned Qwen processor on synthetic and real three-camera/FAST examples; target independence and complete action-code retention verified | Broader source normalization, state/geometry integration, pretrained forward/backward |
| A1.5 | Pin and integration plan | No native inference | Native policy/expert adapter and checkpoint transfer |
| Tau0 | Pin, proposal client, world commands, export audit, unqualified native inference probe, CPU environment recipe | Full LIBERO export audit; native model imports; 23 upstream adapter/wire/evaluation tests; real A2D preprocessing | Matched LIBERO fixture and native inference; qualify CUDA dependencies and services |
| Visual goals | Future-pair export, goal provenance guard, cache, generator jobs | Synthetic-image tests/exports | Actual generated images and downstream success study |
| Fast editor | Lazy Flux2Klein Diffusers wrapper | Not imported/executed | Dependency/model revision, zero-shot quality, robotics LoRA and ROCm |
| Training loop | Local reference learner, optional DDP | Single-process CPU and two-process Gloo | Production native learner, FSDP2/Megatron, distributed resume |
| Numerical harness | Hashes, fixed noise/times, forward/sampling/backward/update probes | Exact CPU self-parity | Native captures and CUDA-versus-ROCm checks |
| NVIDIA batching | Preflight, deadline, job logs and process cleanup | Dry-run and preflight unit tests; no GPU | Fill native jobs/assets before booking window |
| Primus | Pinned external dependency and integration plan | Not cloned/installed | Complete and qualify actual training integration |
| BF16/FP8 | Explicit precision paths/guards | Reference CPU FP32 plus CPU BF16-autocast smoke | Actual AMD BF16/FP8 learning and closed-loop qualification |
| RL | Source pins and algorithm boundary | No robot RL | Correct stochastic-policy loss, actors, rewards and ROCm path |
| Real robot | No actuator driver | None | Hardware, safety controller, data and trials |

## Important implementation corrections

- The fully pretrained W0 Base checkpoint is not the same as Wan-initialized ActionDiT weights.
- Matching action dimension or hidden width does not establish matching semantics.
- Tau's world-model search/proposal path is not proof that its native low-level checkpoint accepts an image-goal argument.
- `pai.eef80.v1` is our schema; direct donor checkpoint I/O compatibility is not claimed.
- The reference auxiliary scalar codec is a test instrument, not an implementation of FAST/ActionCodec.
- Native W0 uses the opposite flow sign/time convention to the reference. Its inspected training-weight function also differs from the reference unweighted MSE.
- The current native composition may perform two Qwen forwards for separate prefix and CE paths; this is correct but not yet optimized.
- The reference training checkpoint saves optimizer/RNG state, but exact multi-worker distributed data-resume semantics are not implemented.
- No generic image-editing metric proves goal reachability, and no oracle-goal loss proves goal-free improvement.

## Delivered artifacts

`artifacts/` contains actual local execution logs/reports. The small `.pt` and `.npz` files were generated by this workbench and are explicitly synthetic. They contain no pretrained third-party weights or robot benchmark trajectories.

The repo has no hidden network calls on import. Native dependencies are loaded only when explicitly requested. Unsupported paths fail with actionable errors rather than substituting another model/backend.
