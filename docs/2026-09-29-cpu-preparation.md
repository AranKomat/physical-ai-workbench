# CPU preparation for native reproduction

GPU rental is deferred at the user's request. No previous host was contacted.

## Measured results

- Complete Tau LIBERO export downloaded at revision
  `ce1dd8f0d917011ce45671fa40edaae1fd6a6ec1`.
- Pinned checksum manifest and all 13 listed files verified, including weights,
  tokenizer, normalization and data transforms. Header inspection found 893
  tensors, 3,015,404,136 stored elements, all BF16. Weight file: 6,030,951,200 bytes.
- Native contract checked: 40D model state/actions, horizon 10, active slots
  0 through 8 and 18; two cameras; EEF representation. No inference yet.
- Bundled A2D sample audited: 25 episodes, 9,469 frames, 30 Hz, 163D state,
  36D action. All state/action values finite; per-episode frame indices contiguous
  and timestamp intervals consistent. There are 9,244 complete 10-frame windows.
- This sample is not LIBERO. These windows have not been converted into training
  examples; video decoding, synchronization and source-native conversion remain.
- Local regression suite: 101 tests passed.

Reports: `artifacts/local_20260929/tau0-export-audit.json` and
`artifacts/local_20260929/tau0-sample-audit.json`. Weights and source data are
ignored locally, not redistributed through this repository.

## Native environment issue

An isolated `.venv-tau` was created with Python 3.12, torch 2.7.1,
torchvision 0.22.1 and Transformers 5.5.4. This is a CPU preparation environment,
not a qualified inference runtime or a complete reproduction of upstream pins.
The probe CLI help works. Importing upstream `deploy.policy` fails because its
data module eagerly imports the absent LeRobot package.

A resolver dry run with both upstream versions establishes a packaging conflict:

- `lerobot==0.4.1` requires `huggingface-hub>=0.34.2,<0.36.0`.
- `transformers==5.5.4` requires `huggingface-hub>=1.5.0,<2.0`.

No downgrade was applied. Next, inspect upstream's documented installation
workaround or isolate the LeRobot dataset dependencies with an explicitly
recorded override, then run native adapter and import tests. Do not substitute
mock modules or silently downgrade Transformers. This packaging issue is not
a failed pretrained-policy inference.

## Prepared commands

```bash
.venv/bin/python scripts/audit_tau_export.py weights/tau0-libero \
  --output runs/tau0-export-audit.json
PYTHONPATH=src .venv-tau/bin/python scripts/audit_tau_sample.py \
  third_party/tau0/example_data --output runs/tau0-sample-audit.json
```

`scripts/probe_tau_native.py` is prepared but unqualified until native imports
and a matched fixture work. It verifies the clean source pin and full export,
accepts explicit camera/state arrays, invokes native encode/infer/decode and
records actions/provenance. A finite action chunk is not closed-loop success.
Its NPZ input requires two already-native-processed 224x224 RGB images and an
8D LIBERO state; do not feed it A2D data. Prefer a real simulator capture for
qualification, and preserve the task/start/observation provenance separately.

## Hardware and order

One RTX 4090 is the intended initial native-inference and CUDA-reference host.
The 6 GB weight footprint supports this choice but peak memory has not been
measured. Full-model training needs weights, gradients, optimizer states and
activations; do not promise that it fits 24 GB.

AMD is needed later for ROCm inference, sampler and gradient parity as well as
training. It is not inherently a training-only GPU. Do not rent both platforms
continuously before local fixtures and jobs are ready.

Remaining order: resolve native runtime; prepare matched LIBERO observations;
reproduce native policy; audit research donor transfer and real action labels;
capture CUDA forward/sampling/gradient/update references; qualify AMD BF16;
only then consider larger training and FP8. CPU audits are not physical results.
