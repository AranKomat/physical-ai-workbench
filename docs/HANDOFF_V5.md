# Physical-AI Workbench — Research and Implementation Handoff v5

**Date:** 2026-09-29  
**Purpose:** Continue the small-team physical-AI research project and a separate near-term startup demonstration track.  
**State:** CPU-tested implementation workbench; native pretrained-model integration and GPU qualification remain unfinished.  
**Read first:** `README.md`, `docs/IMPLEMENTATION_STATUS.md`, then `docs/NEXT_AGENT.md`.

This version incorporates the decisions made after the v4 discussion: preserving pretrained knowledge, compact subtask reasoning, optional visual goals, Tau0 as an open hierarchical reference and independent demonstration substrate, and a bounded NVIDIA correctness session. It also describes exactly what has now been implemented.

## 0. Recovery and provenance

The immediately preceding assistant message said a repository and v5 handoff had already been saved. In this resumed workspace, they were **not present**. Only the original v1 handoff and the two discussion/context Markdown files were mounted. This implementation was rebuilt in the present run; it is not a recovered, GPU-validated earlier codebase.

The original handoff establishes the project's longer-term intent: a strong, reproducible model artifact to build physical-AI expertise, credibility and deployment infrastructure, not a permanent bet on owning the world's best robot foundation model. Later conversational decisions override obsolete v1 choices such as a randomly initialized motor head or a mandatory 4B/9B/27B scaling sweep.

Direct `git ls-remote` failed with `Could not resolve host: github.com`. The GitHub connector did work. Relevant source interfaces and immutable repository revisions were inspected through that connector. `upstreams.lock.json` contains those revisions. The bundle does **not** contain cloned upstream repos or pretrained weights. `scripts/bootstrap_upstreams.py` fetches the pinned source on a networked host, with a dry run by default.

Source-derived facts, our proposed adaptations, and measured results must remain separate. Old chat benchmark numbers are not newly reproduced evidence. This handoff deliberately avoids repeating unverified cross-paper leaderboard rankings.

## 1. Two tracks, one useful technical foundation

### Track R — research/model artifact

Build a capable open multimodal brain directly coupled to a pretrained continuous motor model. The question is:

> How much does the capability of the pretrained multimodal brain improve semantic, long-horizon and out-of-distribution physical behavior, when the action representation and motor prior are held as fixed as practicable?

The target architecture remains:

```text
current multi-view observations + bounded history + task
                  + optional concise subtask / grounding
                                   |
                            pretrained Qwen
                                   |
                      retained token/layer states
                                   |
                 trainable conditioning bridge / layer mixer
                                   |
                     pretrained continuous flow motor
                                   |
                explicit, physically aligned action interface
                                   |
                        independent robot controller
```

Use public robot data and reproducible simulation evaluations. AMD/Primus engineering, checkpoint-transfer diagnostics, numerical qualification and a transparent model release are part of the artifact. Do not claim a new architecture merely because independently known components were combined.

### Track D — startup/real-world demonstration

Use the **intact native Tau0 policy** as the first demonstration substrate, with its native normalization, action interface, proposal model and optional world-model tools. This track need not wait for our research checkpoint and need not use that checkpoint.

Choose accessible, reliable hardware, preferably ordinary arm/parallel-gripper manipulation before high-DoF hands or a humanoid. Expect camera/embodiment calibration and possibly target-specific post-training. An impressive in-distribution demonstration by an upstream group is not evidence of zero-shot transfer to arbitrary hardware.

The useful demonstration is semantic choice, task decomposition, observation of failure, correction and verified completion—not simply showing that an upstream checkpoint can move an object. State clearly which model and data are inherited and which parts we built. Do not hide the pretrained-policy lineage.

Hardware is not available here. No hardware bridge, safety certification, robot trial, or real-world success is included in this bundle.

## 2. Source and component ownership

| Component | Role | Boundary |
|---|---|---|
| Qwen-RobotManip | Physical alignment and broad-training design reference | No claim to reproduce its unreleased complete training system |
| InternW0-Delta | Data adapters, source catalogue, pretrained Base ActionDiT candidate | Extracting its motor from Wan/MoT is a transfer experiment, not lossless reuse |
| InternVLA-A1.5 | Qwen-native implementation patterns and pretrained comparator | Layer-coupled expert cannot be swapped between arbitrary Qwen depths without adaptation |
| Tau0-VLA | Native demo policy; proposal/memory/world-model scripts; optional third donor | Native 40D policy and our 80D interface are not interchangeable |
| Primus | Pinned ROCm environment and future distributed/FP8 integration substrate | This bundle does not yet implement a qualified Megatron VLA backend |
| Miles | Later token-policy RL reference | Not a continuous-action RL drop-in |
| Miles-Diffusion | Later flow/diffusion-RL reference | Robotics, rewards, likelihood treatment and ROCm path remain to qualify |
| PI knowledge insulation | Separate VLM representation learning from motor-loss gradients | Stop-gradient alone is insufficient; the VLM needs its own robot-relevant objective |
| Fast image editor / Tau world model | Optional visual desired-state generator | Generic image-editing quality does not establish reachable robot goals |

Do not turn our repo into a monolithic fork of Primus, Tau0 or InternVLA. Keep donor code behind explicit adapters and preserve their own environments until the shared path is qualified.

## 3. Pretrained motor decision

### Primary comparison

The first serious small-budget comparison remains **A1.5 pretrained expert versus W0-Delta Base ActionDiT**. Tau0 may enter as a third candidate only if it integrates easily or the first two transfer badly. Do not launch three full foundation runs.

Start by running an upstream checkpoint in its native contract. Only then change the representation or conditioning. Otherwise a bad result cannot be attributed to the new architecture rather than an already broken loader.

The candidates have different architectures. Calling their comparison a pure initialization ablation is inaccurate. It is a comparison of **donor systems and transfer recipes**, including adaptation cost. Hold Qwen choice, data splits, target semantics, budgets and evaluation constant where possible; disclose what cannot be held constant.

### W0-Delta specifics verified in this run

At the pinned revision:

- `docs/models.md` lists `InternW0-Delta-Base` for pretrained-policy reuse.
- `ActionDiT` has a standalone `forward(action_tokens, timestep, context, context_mask)` interface.
- `configure_vlm_conditioning(context_dim)` preserves action-side Q/output projections and query norms while replacing cross-attention K/V for the new context width.
- The action constructor in `configs/model/wam.yaml` uses hidden width 1024, FFN width 4096, 30 layers, 24 attention heads with head dimension 128, text dimension 4096 and frequency dimension 256. These numbers do not imply that attention inner width equals hidden width.
- The native scheduler forms `(1-sigma)*data + sigma*noise`, predicts `noise-data`, and uses timesteps scaled by 1000. The inspected action config uses train/inference shift 1.0. Its training weighting is not identical to the unweighted reference loss in this workbench.

Our reference flow convention is the reverse: `t=0` noise, `t=1` data, target `data-noise`. The native wrapper therefore requires an **explicit** time direction, scale and velocity sign. For the inspected unshifted source, the candidate mapping is native time `1000*(1-t)` and output sign `-1`; it still needs numerical qualification with the actual checkpoint.

Do not use the upstream `from_pretrained` helper without checking what it loads. Its Wan-backbone initialization payload is distinct from the fully pretrained W0 Base checkpoint. This bundle's loader refuses missing/unexplained tensor keys rather than silently falling back to random weights.

### Preservation schedule

Train new interfaces first; then conditioning/norm layers; only then full motor weights at a lower learning rate if validation benefits. New interfaces include changed action encoders/decoders, Qwen-context projections and any new state/history adapters.

Completely freezing a motor is not mathematically impossible when input semantics change: sufficiently expressive adapters could translate them. But it is an empirical capacity/optimization question, not a guarantee. Similarly, full fine-tuning does not inevitably erase the prior. The staged schedule is a stability-oriented default to verify.

LoRA is optional for preserving the motor prior, not mandated by its memory footprint. The workbench includes a tested zero-initialized LoRA linear residual and reference staged-freezing utilities. Native donor parameter groups still need explicit review; do not reuse reference layer-name rules blindly.

## 4. Backbone capability, not a scaling law

Preserve the agreed candidate roles:

- Qwen3.5-2B for inexpensive integration and first controlled runs.
- Qwen3.6-35B-A3B as a possible intermediate capability/cost point.
- Qwen3.8-27B as the requested stronger dense candidate.
- Qwen3.5-9B as a simpler dense fallback.
- Flash-Next only as a later memory/throughput qualification, not a dependency.

Only the Qwen3.5-2B public config was freshly fetched successfully in this run. It declares `qwen3_5`, hidden size 2048 and 24 layers. The attempted Qwen3.8 config fetch failed here; this is not proof that the model is unavailable. Resolve and inspect it on the training host. `configs/models.yaml` distinguishes requested candidates from successfully inspected configs.

Same Hugging Face model class does not establish identical layer schedules, fused kernels, optimizer support, checkpoint conversion, tokenizer or image preprocessing. Inspect configs and run actual forward/backward tests. Do not infer tensor/KV compatibility from matching hidden width alone.

Compare equal consumed tokens and equal node-hours. The outcome of interest is semantics/OOD/recovery, not a parameter-count exponent or merely lower imitation loss.

## 5. Knowledge insulation: adopted hypothesis, precisely implemented boundary

PI's method trains the VLM with discrete action-token cross-entropy and auxiliary vision-language supervision while stopping the continuous motor loss from backpropagating into it. It does **not** mean simply freeze Qwen forever, nor does a separately trained codec alone establish the same gradient boundary. [S7]

Use this structure:

```text
observations -> Qwen -> action-token / planning / VL CE
                   |
                stop-gradient
                   |
          trainable bridge / layer mixer
                   |
           continuous flow expert -> flow loss
```

The detach goes **before** the trainable conditioning bridge. The bridge must still learn from motor loss. Also insulate any alternative motor gradient routes, such as shared embeddings or a geometry adapter attached upstream of the detach.

`ReferenceVLA` implements and tests this boundary with an auxiliary autoregressive scalar-token decoder. That codec is deliberately a **diagnostic surrogate**, not FAST, ActionCodec, PI's full KI recipe, or evidence of equivalent convergence. The native Qwen wrapper accepts separate assistant-only CE batches so real FAST or another validated, frozen action tokenizer can be supplied.

Production work still needed: select/version the actual auxiliary tokenizer, prepare labels from our aligned actions, verify teacher-forcing masks, attach the CE path to the Qwen LM head, and profile its cost. Runtime actions remain continuous; auxiliary discrete targets need not be generated at deployment.

**No target leakage:** teacher-forced action tokens must not appear in the context supplied to the motor. The native composition uses separate `prefix_inputs` and `auxiliary_inputs`. The reference objective recomputes a goal-free prefix for auxiliary CE when privileged goal images are present.

The often-mentioned 7.5x figure comes from PI's broader KI recipe comparison, not a guarantee that `detach()` alone makes our training 7.5x faster. A short joint-versus-insulated check is worthwhile; do not recreate a giant full-factorial study.

Maintain a held-out preservation suite: object recognition, referring-expression grounding, OCR, spatial relations and general VQA. Good motor loss does not establish retained Qwen capability.

## 6. Representation and data contract

### Do not confuse three layers

1. Physical meaning: frames, units, EEF rotations, gripper conventions and action reference.
2. Storage schema: which named channel occupies which array slot and mask.
3. Decoder/tokenizer: continuous flow, auxiliary scalar/FAST tokens, or a later codec.

An 80D array is not inherently physically aligned. Our `pai.eef80.v1` is explicitly an **internal research schema**, not a byte-identical reproduction of any donor's 80D layout. Native W0, A1.5 and Tau0 mappings must be audited separately.

Current workbench slots:

```text
0:6     left EEF: translation delta + spatial rotation vector
6       left gripper
7:13    right EEF
13      right gripper
14:17   base channels
17:37   left-hand reserved channels
37:57   right-hand reserved channels
57:80   explicit source-native fallback channels
```

Unused dimensions remain zero and are masked from targets and sampled noise. A controller receiving native fallback slots must know their named interpretation; never combine incompatible native joint ontologies without embodiment/schema metadata.

### Geometry

`geometry.py` defines `T_camera_from_base` as a point transform. EEF-origin displacement rotates by the camera rotation; a spatial rotation increment is computed with `log(R_target R_current^T)`, not Euler-angle subtraction. Hold a moving wrist-camera reference fixed at the anchor observation when expressing a future chunk. This is distinct from applying an adjoint to an origin-referenced spatial twist.

Specify whether each future command is relative to the previous target, the anchor state, or a measured contemporaneous state. The numeric values are not interchangeable. Source adapters must perform the intended conversion, not merely label it.

### Tiered alignment

Tier A: calibrated camera-frame EEF.  
Tier B: explicit robot-base/task-space EEF.  
Tier C: explicit native joint/action fallback.

Never invent calibration or FK. Preserve Tier B/C data where useful, but report coverage separately. Joint versus EEF results from other papers do not establish that one representation is universally best; our alignment default is a project hypothesis with strong prior motivation, not dogma.

Global physical scales are shared across aligned semantic channels. No per-robot quantile normalization that makes the same meter displacement mean different targets. Native fallback joints may use separate named conventions. State normalization and action-displacement normalization are not automatically the same.

### Dataset implementation boundary

The workbench reads a local `episode.json + arrays.npz` interchange with explicit timestamps, cameras, arrays, masks, subtask segments, provenance and calibration. It includes a Dataset/DataLoader path and can train the reference model from these episodes.

It does **not** implement every upstream public dataset converter or scalable LeRobot video decoding. Use W0's source adapters as the main catalogue, audit them, then export this contract or replace the local Dataset with an equivalent streaming adapter. The NPZ cache is for modest integration subsets, not tens of thousands of hours.

## 7. Corpus strategy and quality metadata

Follow the W0-Delta catalogue and mixture as a starting prior rather than inventing a bespoke collection. The discussed outer mixture is about 90% robot examples and 10% general VL examples. The discussed within-robot family prior is 80/10/8/2 robot/H2R/UMI/ego, subject to source availability and actual valid action supervision.

**Raw ego without actions belongs in an auxiliary visual/language stream, not an all-zero fake action batch.** Missing families are reported and weights renormalized. The simple reference sampler is family-balanced and uniform over episodes within a family; it is not a claim to exactly reproduce all W0 weighting rules.

Keep dataset revision, original physical episode ID, conversion version, and license/provenance metadata. Annotation overlays such as RoboInter must share a split group with their underlying DROID/RH20T episodes. Do not count them as new independent robot hours. Deduplicate before train/validation/test splitting.

Keep quality, mistake, control mode and speed only when their semantics are reliable. Unknown quality is `null`, not “5.” Smoothness is not success. Dataset duration alone is not a universal motion-speed measure. Remove corrupt/misaligned data; potentially retain useful failed/recovery segments with honest labels rather than train them as uniformly desirable actions.

Implemented checks cover finite values, shapes, masks, monotonic timestamps, missing cameras, calibration matrices, split overlap, static/frozen-data diagnostics and explicit thresholds. Language-video consistency, accurate FK checks and calibrated quality scoring still require source-specific or teacher-assisted work. Do not label the current numerical validator as a complete high-quality curation system.

Available corpus and tokens consumed by one run are separate quantities. A 1K–3K-hour pass remains plausible but is not a hard cap. Choose size from measured throughput, coverage, number of passes and diversity.

## 8. Compact reasoning and memory

Default output: one concise semantic subtask, e.g. “Open the microwave door with the left arm.” Add target grounding or a short action hint only when useful. Avoid long prose before every action chunk.

The Tau proposal interface supports `subtask_only` and `full_qa` (`think`, memory, subtask), with head/left/right images. Our HTTP payload builder follows that source contract. It is not a motor controller. [S4]

Refresh when the task changes, verified completion occurs, an execution failure is observed, the scene changes materially, or a fallback timeout is reached. The timeout is not a claim about mean subtask duration. `subtask_duration_summary` measures the distribution from available segment annotations.

Separate **proposed action** from **verified completion**. `ExecutionMemory` never marks a proposed subtask done. Invalid planner parsing must not advance memory. In native Tau mode its model-generated memory remains an observation/claim to verify, not ground truth.

Borrow the memory-corruption training idea as an optional low-cost augmentation: make lagging or falsely optimistic histories and train the planner to correct them from current observations. Keep training-only future labels and corruption machinery out of runtime.

## 9. Grounding, layer routing and fast feedback

### Target crops

Qwen may propose a normalized box and deterministic code may crop the original image. No SAM dependency is required initially. The workbench validates boxes, preserves coordinate conventions and computes IoU. A supplied crop helping a policy does not prove self-grounding works; compare oracle crop, self crop and no crop separately.

Offline labels may come from a stronger model, but record the teacher, prompts, revision and confidence. Keep a human-audited held-out set. Evaluate grounding before and after robot adaptation; do not assume KI completely preserves it.

### Multi-layer conditioning

The workbench contains a simple learned layer mixer. It is **not** a replication of LayerRoute or M²-VLA. Begin with a few selected Qwen layers and compare with final-layer-only only if interface training is stable. Preserve full relevant token states instead of assuming an arbitrary pooled vector carries all semantic and spatial information.

### Fast proprioception

The motor should receive current state and, when available, short state/action history while slower semantic context is reused. Proprioception can reveal some contact effects, but without sufficient sensing/compliance it cannot guarantee observation of slip, object pose or force. Do not drop vision universally or represent inferred contact as a measured tactile signal.

High-DoF hand synergies, tactile inputs, reachability/progress critics and specialized future-feature supervision remain second-wave experiments. They must not block a reliable parallel-gripper policy.

## 10. Visual subgoals: revised, deployment-aware plan

The world-model branch is optional but no longer dismissed merely for being elaborate. The correct question is whether generated goals improve **deployable closed-loop success per latency/compute**, not whether conditioning on a known future makes the training loss easier.

### Four distinct settings

| Setting | Training | Evaluation | Interpretation |
|---|---|---|---|
| G0 | no goal | no goal | deployable baseline |
| G1-oracle | some real future frames + dropout | true future supplied | privileged diagnostic, not a deployment result |
| G1-no-goal | same G1-trained policy | no goal | tests whether privileged training helps the goal-free policy |
| G2-generated | generated or appropriately mixed goals + dropout | generated goals | actual deployment-facing test |

A lower G1 conditional loss is expected because the target contains information about the demonstrated future; it is not proof of a better ordinary policy. Report all relevant modes separately.

“25% goal-conditioned examples” is a sampling choice inspired by the discussion, **not** a requirement to generate images for 25% of all unique frames or episodes in advance. The split between real and generated goals is a separate parameter.

The workbench selects real future pairs within the same episode, split and subtask. Those pairs can train an editor with no teacher image-generation cost. They remain marked `real_future_privileged` and are rejected in deployment mode. Runtime generated goals require generator identity/revision and camera provenance.

### Candidate generator order

1. Tau's released Step1X-Edit-v1p2 robotics LoRA as the domain-adapted reference.
2. A fast few-step editor, initially the requested FLUX.2 Klein family, tested zero-shot and optionally LoRA-adapted.
3. Other editing candidates only if these fail on relevant robot-state metrics; do not survey dozens of generic image leaderboards.

The current Tau guide recommends additional domain adaptation and defaults to 28 denoising steps. Its fine-tuning input is a current image, target image and concise subtask. This is verified source behavior, not a speed claim on AMD. [S5]

The Klein adapter is optional and unexecuted here. Confirm the installed pipeline, checkpoint revision, editing interface, multi-view contract, and whether a robotics LoRA transfers to a distilled few-step checkpoint without degrading useful control. Do not assume a Base-trained LoRA guarantees good four-step robotics results.

Start with a small held-out robot transition set. Test scene/camera preservation, target identity/state change, robot morphology, spurious edits, failure cases and downstream action success. Pixel similarity alone is inadequate because multiple valid future states may exist. No “required amount of fine-tuning” is known; measure a small adaptation curve and stop early if a zero-shot model is sufficient.

### The important Tau versus PI distinction

Tau's released high-level world-model tool predicts a goal image from a head-camera image and subtask. The proposal can feed that tool or a low-level policy. This does **not** establish that Tau's stock low-level checkpoint consumes goal images directly like a PI-style goal-conditioned motor policy.

Keep two modes separate:

```text
Tau-style: imagined outcome helps select a TEXT subtask -> native low-level policy
Our extension: generated IMAGE is an extra input to a policy TRAINED to use it
```

Adding an image argument to an RPC cannot create that learned conditioning ability. Multi-view goal consistency is another independent requirement; a single head-view editor is not automatically a three-camera world model.

### Runtime and cost

Run a generator asynchronously/selectively at subtask boundaries, uncertainty or failure only when justified. Old goals expire on relevant state/task changes; cache keys include the source image, instruction, camera, generator revision, adapter, seed and settings. A cache cannot safely reuse a goal merely because the subtask text stayed the same.

Generated goals describe intended or likely states, not a verified simulator or reachable-state certificate. A planner and critic can share the same hallucination. Preserve collision, motion and hardware checks outside them.

For offline cost, count unique requests, views, seeds, training reuse and generator batching. Use measured GPU-seconds per accepted goal, not a latency number from another model on four NVIDIA GPUs. The workbench's cache and budget utilities expose these quantities; no images were generated here.

## 11. Tau0 demonstration track

Keep its native state/action dimensions, rotation representation, normalization, horizon and controller conventions intact for initial evaluation. Do not impose our research 80D layout on the first real-world demo.

The pinned high-level code provides native proposal inference/serving/LoRA commands and world-model inference/fine-tuning commands. The bundle includes the proposal HTTP client and command builders, but no weights or live inference result.

First reproduce a native simulation or recorded-observation inference example. Then select accessible hardware/cameras and a bounded set of tasks. Collect target demonstrations only as necessary; the required amount is unknown. The upstream model's internal pretraining data is not supplied with this workbench. Hardware trials can evaluate a model without a giant pre-existing dataset, but require scenes, task definitions, resets, safety and outcome measurement.

For a capability comparison, hold Tau's low-level policy fixed and compare its native proposal model with a stronger proposed planner under equal observations and a bounded tool/compute budget. This is different from swapping the brain inside the end-to-end research VLA; report them as separate experiments.

Do not use an oracle simulator state, ground-truth future image, manual crop or human-provided subtask without labeling that assistance. Preserve failures in logs and videos, not only successful hero takes.

## 12. AMD, Primus and precision

Pinned Primus source is an environment/infrastructure dependency. The current repository provides a portable PyTorch reference learner and a tested two-process CPU/Gloo run. It does **not** yet implement or validate native Primus Megatron/FSDP2 VLA training.

The native integration path is:

```text
pinned Primus ROCm environment
    -> install local robotics package without replacing ROCm torch
    -> native Qwen and donor BF16 forward/backward
    -> exact checkpoint/schema/time-contract parity
    -> short supervised training
    -> sharding/distributed checkpoint/resume
    -> qualified FP8 optimization
```

Do not treat a diffusion model config as proof of our custom model's training correctness. Do not assume a quantized inference checkpoint is an FP8 training recipe. FP8 matmuls, master weights, gradients and optimizer-state precision are distinct choices.

Our probe refuses a fake `precision=fp8` cast. A real FP8 path needs the appropriate backend integration and a matched BF16 learning/closed-loop comparison. If qualification fails, retain BF16 rather than declaring the hardware unsuitable or silently relaxing tolerances.

## 13. Concentrated NVIDIA qualification

NVIDIA access is intermittent. **Do all preparation first**, then batch reference work into one focused access window, illustratively up to two hours. This is a resource envelope, not a promised execution time or a requirement for a B200/H200.

Before the window:

- clone pinned source and obtain exact weights on a networked machine;
- prepare fixed processed batches, noise and flow timesteps;
- verify image processors, masks, normalization and checkpoint hashes;
- prepare native-module capture scripts and expected memory requirements;
- dry-run every command and make all files local;
- set `ready_for_cuda` only when the actual native jobs are ready.

During the window, collect the unmodified native checkpoint's forward values, selected activations, sampler trajectory, loss, selected gradients and one-step updates; run a short fixed-data reference curve where it fits. Capture the original W0 conditioning at the actual native boundary before comparing the port. A standalone extracted expert versus itself does not establish equivalence with the full MoT policy.

Then compare AMD BF16, followed by AMD FP8 qualification. Keep final foundation runs AMD-only unless an unexplained discrepancy justifies another focused session.

The portable fixture/probe machinery is implemented and exact CPU self-parity was measured. It is a **reference-model** harness, not a native checkpoint qualification. Native capture adapters are still needed.

Compare tolerances, relative error, gradient norms/cosine, action trajectories and physical-unit errors. Do not demand bitwise BF16 identity across hardware. Do not adjust thresholds until a broken model passes. Repeat within-backend baselines to characterize variation, and add a small closed-loop task check; numerically close open-loop actions can diverge near contact.

`scripts/run_nvidia_window.py` preflights required files/hashes, enforces a total wall budget, logs each job, stops on failure and terminates timed-out process groups. The template is deliberately not marked ready because native fixtures have not been collected.

## 14. RL boundary

RL remains downstream of a strong supervised policy. Miles is the later token/VLM RL reference; Miles-Diffusion is a flow/diffusion RL reference. Their existence does not establish plug-and-play support for our camera/action schema, rewards, simulator actors, native checkpoint or AMD kernels.

A deterministic flow sampler has a distribution induced by its input noise, but it does not expose an inexpensive categorical action log-probability. Do not reuse language GRPO code by treating flow-MSE or a predicted velocity as a log probability. Choose and verify an actual stochastic flow/SDE or other justified policy update method.

Begin RL in BF16, initially freeze most Qwen, compare with the identical SFT checkpoint, and monitor KL/retention, success and physical failure modes. FP8 RL, joint reasoner/motor RL, ActionCodec at runtime, PRTS integration and high-DoF synergies are deferred. The delivered code does not implement robot RL.

## 15. Budget accounting after KI and optional goals

The user's 2B tokens per node-day figure referred to the planned **27B regime**, not a measured rate or a limit for the 2B model. Actual rates must be measured with images, state/history, motor compute and any auxiliary objectives.

For a planning illustration, 9 images at 256x256, about 64 merged visual tokens/image, plus 150 text tokens gives 726 context tokens/window. At one window per source second, that is 2,613,600 context tokens per unique source hour for one pass. This arithmetic excludes extra CE action tokens and does not predict FLOPs or throughput.

Knowledge insulation changes the budget: the backbone also processes action-token targets. The native reference composition currently uses separate prefix and CE forwards for correctness, so duplicated vision/backbone work must be measured. A fused implementation may reuse work later only if target leakage is prevented.

Also count optional goal images and crops, selected-layer storage, repeats of action-noise examples, and state-history updates. One 80D action vector is not 80 Qwen output tokens in the continuous decoder, but the motor still consumes compute.

A 90/10 **example** mixture is not necessarily a 90/10 **token** mixture. The budget module includes that conversion. Report unique physical hours, sampled windows, consumed tokens, passes and node-hours separately.

## 16. Experiment bundles and stop rules

Do not run every idea as a full factorial ablation.

| Bundle | Main question | Promotion rule |
|---|---|---|
| E0 substrate | Can data, gradients, masks and reference training behave correctly? | CPU tests and smoke first; native GPU parity next |
| E1 native correctness | Does the exact donor behave correctly on AMD? | No unexplained forward/sampling/update mismatch |
| E2 motor transfer | Which donor survives our interface with least loss/cost? | Short curves for A1.5/W0; Tau only if justified |
| E3 protection | Does KI-style adaptation preserve Qwen/motor capability? | Short joint-training control; retain useful protection |
| E4 cognition | Do compact subtasks/history and selected-layer states help semantic OOD? | Small targeted tasks, not easy benchmark saturation |
| E5 goals | Do generated visual goals help deployable policy performance? | G0/G1-oracle/G1-no-goal/G2 reported separately |
| E6 data | Quality/diversity versus repeated volume? | Equal-token comparison and limited token-scale curve |
| E7 brain | Does stronger pretrained capability justify node-hours? | Same selected motor/representation/eval where possible |
| D0 demo | Can intact Tau plus planning solve a useful real task? | Independent native setup and safe hardware qualification |
| R0 later RL | Does a verified update improve a strong SFT policy? | BF16, bounded tasks, clear failure/retention metrics |

Defaults with validation rather than broad reinvention: physically explicit representation, broad VL replay, concise subtasks where available, bounded history, honest metadata and preservation of expensive pretrained components.

Conditional additions: multi-layer routing, self-crops, generated goals, corrupted-memory training. Deferred: heavyweight world-model pretraining, many-codec sweeps, joint high-DoF control, full RL, Flash-Next optimization and large sensor suites.

## 17. Repository map

```text
src/physical_ai/
  schema.py, geometry.py       explicit physical action contracts
  data.py, dataset.py          episodes, windows, splits, masks, future pairs
  models.py, flow.py           runnable tiny reference VLA / flow objective
  native.py                    lazy Qwen + native W0 adapters, unqualified
  training.py                  gradient policy, stages, LoRA, smoke run
  checkpoints.py               fail-closed loads, atomic reference checkpoints
  planning.py                 concise subtasks, verified memory, crop helpers
  world.py                    provenance-safe goals, cache, editor/Tau bridges
  validation.py               fixed oracle probes and comparison reports
  budget.py                   transparent accounting arithmetic
  cli.py                      local commands
scripts/
  bootstrap_upstreams.py      pinned source fetch, dry-run by default
  resolve_hf.py               immutable model config/provenance resolution
  hf_probe.py                 actual local Qwen forward/backward probe
  train_reference.py          local-window reference learner, optional DDP
  run_nvidia_window.py         concentrated CUDA job runner
  validate_cpu.sh              reproducible offline validation
configs/                      project decisions and experiment specifications
artifacts/                    actual CPU reports and explicitly synthetic fixtures
```

## 18. What still requires coding, not merely GPU execution

Be explicit about these remaining tasks:

1. Native A1.5/Tau expert/full-policy adapters beyond the existing upstream commands and Tau proposal bridge.
2. Native W0 checkpoint extraction key paths, shape audit and original-conditioning capture; no weights have been examined locally.
3. A production HF collator and real FAST/equivalent target preparation with assistant-only loss and no motor-target leakage.
4. Large-scale LeRobot/video streaming adapters and source-specific camera/FK/normalization conversions.
5. A real Primus/Megatron or FSDP2 integration, optimizer precision choice, distributed checkpointing and exact resume. CPU/Gloo DDP does not prove these.
6. Native CUDA/ROCm parity capture hooks, followed by actual hardware execution and closed-loop evaluation.
7. Simulator adapters and target-robot drivers with independent safety enforcement.
8. Generator inference/LoRA qualification; goal-conditioned policy integration beyond the reference path; optional high-level search/value scoring.
9. Robot RL and its rollout/learner consistency tests.

These are bounded next-agent tasks, not hidden successes in this deliverable. The reference model should not be scaled up and represented as a pretrained donor replacement merely because it runs.

## 18A. Validation executed for this release

The rebuilt workbench passed **93 CPU tests** with no skips or failures. A tiny fixed-batch reference model completed 40 FP32 training steps, with flow loss falling from 1.023472 to 0.080292. Saving/reloading the checkpoint reproduced its sampled output exactly, and inactive channels remained zero.

The actual canonical episode Dataset was also exercised through an eight-step CPU learner, a separate eight-step CPU BF16-autocast smoke, and a four-step two-process CPU/Gloo DDP run. All used manufactured synthetic episodes, not robot demonstrations. Two deterministic reference captures matched exactly across **219 tensor records**.

Source-fetch commands and the bounded NVIDIA-session plan were dry-run tested. The Python package installed locally in editable mode without replacing dependencies. `docs/VALIDATION_REPORT.md` and `artifacts/VALIDATION_SUMMARY.json` carry the detailed evidence.

None of these results validates native pretrained-model transfer, NVIDIA-versus-AMD parity, FP8, a simulator benchmark, real-world robot behavior, or RL. Those remain explicit next-agent milestones.

## 19. Next-agent immediate sequence

Run the offline validation once. Read `IMPLEMENTATION_STATUS.md` and inspect the actual reports. Clone pinned sources using the bootstrap script. Verify the A-series branch is `master`, not the assumed `main` in several old links.

Next reproduce **one native donor** in its original contract. Inspect the exact checkpoint keys and flow conventions; create immutable manifests. Complete the smallest real Qwen/native-motor BF16 training step with real data before attempting FP8 or optional goals.

Prepare the consolidated NVIDIA window only after the native test jobs are ready. Use its output to qualify AMD. Then run the small motor-transfer/protection experiment and select a stable architecture.

In parallel, use Tau's native proposal/world/model scripts as a separate demonstration route. Do not force our research schema into that path. Obtain actual compatible hardware and target-specific data only when the simulation/native setup is reliable.

After those gates, expand the curated public mixture, compare backbone capability, and add only the optional cognition/world-model components that improve deployable outcomes.

## 20. Sources and verification levels

**Source code freshly inspected in this run:**

- [S1] Primus pinned repository: https://github.com/AMD-AGI/Primus/tree/43e1c9bece1d7a869c4371de7372803808c7021c
- [S2] W0 checkpoint guide: https://github.com/InternRobotics/InternW0-Delta/blob/90801baa3bdc7829c1e4989edfe3d0fb2410ea6a/docs/models.md
- [S3] W0 ActionDiT: https://github.com/InternRobotics/InternW0-Delta/blob/90801baa3bdc7829c1e4989edfe3d0fb2410ea6a/src/wam/model/modules/experts/action_dit.py
- [S4] Tau proposal: https://github.com/sii-research/tau-0-vla/blob/f1665fbaf624d1468b168e3a54cccc9e326212d9/high_level/proposal/README.md
- [S5] Tau world-model guide: https://github.com/sii-research/tau-0-vla/blob/f1665fbaf624d1468b168e3a54cccc9e326212d9/high_level/world_model/README.md
- [S6] A-series repository/branch: https://github.com/InternRobotics/InternVLA-A-series/tree/e6fc904f9edbfb14532e97095fc2372202517f76
- [S7] PI knowledge-insulation explanation: https://www.pi.website/research/knowledge_insulation
- [S8] Qwen3.5-2B config: https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json
- [S9] W0 scheduler: https://github.com/InternRobotics/InternW0-Delta/blob/90801baa3bdc7829c1e4989edfe3d0fb2410ea6a/src/wam/model/backbones/wan22/schedulers/scheduler_continuous.py
- [S10] W0 model constructor config: https://github.com/InternRobotics/InternW0-Delta/blob/90801baa3bdc7829c1e4989edfe3d0fb2410ea6a/configs/model/wam.yaml

**Design references from the discussion; benchmarks and release claims are not reproduced here:**

- Qwen-RobotManip: https://github.com/QwenLM/Qwen-RobotManip
- PI0.7: https://www.pi.website/download/pi07.pdf
- Tau0 paper: https://arxiv.org/abs/2608.16885
- G0.5: https://arxiv.org/abs/2608.11739
- W0-Delta paper: https://arxiv.org/abs/2609.31394
- VISTA: https://github.com/vista-wm/Vista-WM
- FLUX.2: https://github.com/black-forest-labs/flux2
- Miles: https://github.com/radixark/miles
- Miles-Diffusion: https://github.com/radixark/miles_diffusion
- OpenPI: https://github.com/Physical-Intelligence/openpi

Before a public quantitative claim, verify the primary paper, exact benchmark, checkpoint, fine-tuning budget, observation privilege and evaluation protocol. Before redistribution, preserve source notices and check applicable terms. A declared research purpose does not itself grant rights to every dataset or checkpoint.
