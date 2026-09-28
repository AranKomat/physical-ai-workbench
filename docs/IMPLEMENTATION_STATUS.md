# Implementation status

This file is the boundary between executable work and planned integration. A CPU pass is not a pretrained-policy result.

Latest continuation: **123 local tests pass**. See
`2026-09-29-cpu-completion.md` and `CPU_WORK_PLAN.md` for the eight-item preparation
results. Real captures and full checkpoint audits are complete; no pretrained
GPU inference, native research training or closed-loop success is claimed.

Earlier continuation on 2026-09-29: four core pinned sources cloned, 108 local tests passed,
and the real Qwen3.5-2B processor passes multimodal prefix/CE separation checks.
See `2026-09-29-integration.md` for evidence and the first native reproduction plan.
See `2026-09-29-cpu-preparation.md` for the completed Tau weight audit, real A2D
record audit, deferred GPU access and unresolved native dependency conflict.
The follow-up `2026-09-29-native-cpu.md` records a tested, explicitly overridden
CPU dependency setup: native model imports, 23 upstream tests, and three decoded
real A2D windows. GPU inference is still unexecuted.
`2026-09-29-real-action-labels.md` adds real FAST/Qwen label collation and a
synthetic-tested native LIBERO fixture converter; genuine capture subsequently
completed for three task starts with six nonblank images.

| Area | Implemented | Executed here | Remaining |
|---|---|---|---|
| Physical action contract | Versioned 80D research schema, masks, explicit native channel mapping | CPU tests | Audit each donor/source's actual slot semantics |
| Geometry | SO(3)/SE(3), camera EEF delta conversion and inverse | CPU round-trip/edge tests | Source FK, wrist-camera timestamp conventions, native controller audit |
| Data | Validation, split groups, windows, family sampler; native collator with provenance and masks | 50 complete real A2D windows, 25 episodes, 13 reloadable batches through native LeRobot/PyAV and FAST/Qwen | Physically audited research-schema conversion, held-out selection and temporal/geometry audits |
| Quality | Numeric checks, unknown metadata, frozen/static diagnostics | CPU tests | Calibrated semantic/kinematic quality judgments |
| Reference policy | Tiny image/text brain + cross-attention flow motor | Synthetic CPU learning | Not a pretrained VLA |
| KI mechanics | Stop flow gradients before bridge; auxiliary AR loss; checked FAST labels; vocabulary expansion | Gradient/leakage tests; 50 FAST windows; tiny actual Qwen tied/untied expansion and exact resume | Real pretrained CE/flow forward-backward and donor integration |
| Motor preservation | Reference parameter groups/stages, LoRA residual | CPU tests | Native donor-specific grouping and retention evaluation |
| Layer routing | Learned scalar layer mixer | CPU tests | No LayerRoute reproduction or claimed gains |
| Native W0 | Strict extraction/loading; explicit time/sign conversion; preserved pretrained K/V | Full checkpoint SHA audit; all 824 motor shapes match; reduced actual native forward/backward and exact checkpoint roundtrip | Pretrained inference, action semantics and transfer qualification |
| Native Qwen | Lazy HF wrapper, prefix/CE separation, composition with motor | Real pretrained processor; tiny random native vocabulary/resume checks | Actual pretrained backbone forward/backward and memory qualification |
| Native multimodal collation | Observation processing, assistant-only CE masks, independent prefix/auxiliary forwards | Pinned Qwen processor on synthetic and real three-camera/FAST examples; target independence and complete action-code retention verified | Broader source normalization, state/geometry integration, pretrained forward/backward |
| A1.5 | Separate native prefix-cache adapter and isolated patched runtime | Full checkpoint SHA/inventory; reduced random native prefix-cache forward/backward | Full pretrained inference and transfer |
| Tau0 | Pin, proposal client, world commands, export audit, native inference probe, CPU environment recipe | Full LIBERO export audit; native imports; 23 upstream tests; real A2D preprocessing; three genuine LIBERO captures | Native inference/rollouts; qualify CUDA dependencies and services |
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

The latest CPU preparation adds 50 real native A2D windows with FAST/Qwen
labels, reloadable native-schema batches and provenance; tiny actual Qwen
vocabulary expansion/optimizer resume checks; and real LIBERO captures.
These close preparation gaps in the older data/KI/Tau rows above, not their
pending pretrained training or native-policy evaluation. Public receipts under
`artifacts/local_20260929/cpu_completion/` include simulator PNGs and metadata;
large data, checkpoint weights and generated batches remain local.

`artifacts/` contains actual local execution logs/reports. Its small `.pt` and
`.npz` reference files are explicitly synthetic. No pretrained third-party
weights are committed; real simulator imagery is labeled separately.

The repo has no hidden network calls on import. Native dependencies are loaded only when explicitly requested. Unsupported paths fail with actionable errors rather than substituting another model/backend.
