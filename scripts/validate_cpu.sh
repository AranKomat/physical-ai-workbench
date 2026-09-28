#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p artifacts
python -m pytest tests -q --junitxml=artifacts/pytest.xml | tee artifacts/pytest.txt
python -m physical_ai smoke --output artifacts/cpu_smoke --steps 40
python -m physical_ai make-demo-data --output artifacts/demo_data
python -m physical_ai audit-data artifacts/demo_data/manifest.jsonl --output artifacts/data_audit.json
python -m physical_ai goal-pairs artifacts/demo_data/manifest.jsonl --output artifacts/goal_pairs --pairs-per-episode 2
python -m physical_ai make-fixture --output artifacts/golden_fixture
python -m physical_ai probe artifacts/golden_fixture --output artifacts/probe_cpu_a --learning-steps 6
python -m physical_ai probe artifacts/golden_fixture --output artifacts/probe_cpu_b --learning-steps 6
python -m physical_ai compare artifacts/probe_cpu_a artifacts/probe_cpu_b --output artifacts/cpu_self_parity.json --atol 0 --rtol 0
python -m physical_ai budget > artifacts/token_budget.json
python -m compileall -q src scripts
