# Data and visual-goal contracts

## Episode interchange

`Episode` stores synchronized uint8 camera frames, timestamps, action/state arrays, boolean masks, source/provenance, split group, optional calibration and segment labels. The physical action frame and delta reference are explicit.

Our internal `pai.eef80.v1` slot order is documented in `schema.py`. It is not a promise of compatibility with W0's 80D, A1.5's 32D or Tau's 40D layout. An upstream field-to-slot map and physical transform are required. Never infer semantics from shape.

`PhysicalScaler` uses versioned shared per-channel physical scales, with no silent clipping. The bundled unit scaler applies only to synthetic fixtures. Fit/choose real action ranges using train data and preserve them in inference/checkpoint artifacts.

Use alias-aware physical episode groups for splitting. Different annotation exports of one DROID episode must not land in different splits. The code rejects an explicitly detected split-group overlap; accurate alias IDs are still a data-engineering responsibility.

## Future pairs

`pai goal-pairs` creates start/target image pairs from existing episode frames. It does not run a generator. Targets are sampled strictly in the future and within a subtask; missing subtask labels are not silently replaced by an entire long-horizon task label.

Each record includes original episode/split group, camera, timestamps and `goal_kind=real_future_privileged`. Images for the future target belong only to training or an explicitly labeled oracle evaluation. The generator job exporter sends the source image and instruction, never the target path.

## Tau world-model commands

The source-verified input shape is:

```json
{"id":"sample_001","image":"images/start.png","target":"images/goal.png","instruction":"Pick up the cup with the right arm."}
```

`target` is a fine-tuning field, not an inference input. `tau_world_command` builds native inference/fine-tuning commands without a shell. The repo's reported fine-tuning wrapper is used in its own environment; our code does not recreate that trainer.

Tau's high-level search/world-image tool and a motor trained to consume visual goals are different interfaces. Do not infer that its native low-level policy can accept arbitrary generated image goals.

## Cache and editor evaluation

Goal cache identity includes actual input image hash, subtask, camera, generator model/revision, adapter hash, seed and settings. Cache hit rate is not an excuse to use stale goals on changed scenes.

Evaluate accepted physical goal quality and downstream closed-loop outcomes, not only perceptual similarity. A plausible generated image may be unreachable or geometrically inconsistent. Oracle, generated and goal-free results must be separate.

The optional Klein wrapper has not been imported or executed in this environment. Its model quality, few-step LoRA transfer and AMD runtime are unverified. The Tau Step1X adapter is the domain-specialized reference, not automatically the fastest or best option.

## Quality and safety

Numeric validation is not semantic curation. Unknown quality/failure remains unknown. Learned quality labels need provenance and calibration. Reject corrupt data; retain useful recoveries without labeling them as uniformly optimal behavior.

No component in this repo authorizes physical actuation. Use independent hardware limits, contact/force protections where available, stale-observation checks, watchdogs and emergency stop facilities in the actual robot stack.
