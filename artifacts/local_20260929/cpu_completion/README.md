# CPU preparation evidence

These are measured local receipts, not pretrained robot success results.

- `libero/`: real simulator PNGs/provenance, three task starts, no policy.
- `*-weight-audit.json`: full checkpoint hash and tensor metadata audits.
- `w0-architecture-match.json`: actual Base shapes against native meta model.
- `w0-native-cpu.json`, `a15-native-cpu.json`: reduced random native modules.
- `qwen-vocabulary.json`: tiny random Qwen expansion and save/resume.
- `real-batches.json`: 50 real native A2D/FAST windows, not held-out evaluation.
- `pytest.txt`, `cpu_self_parity.json`: local tests and reference self-parity.

Large weights, raw observations, generated batches and local environments are
excluded from Git. See `docs/2026-09-29-cpu-completion.md` for interpretation,
environment deviations and hardware-dependent work still outstanding.
