#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
validation_dir="${1:-artifacts}"
mkdir -p "$validation_dir"
python -m pytest tests -q --junitxml="$validation_dir/pytest.xml" | tee "$validation_dir/pytest.txt"
python -m physical_ai smoke --output "$validation_dir/cpu_smoke" --steps 40
python -m physical_ai make-demo-data --output "$validation_dir/demo_data"
python -m physical_ai audit-data "$validation_dir/demo_data/manifest.jsonl" --output "$validation_dir/data_audit.json"
python -m physical_ai goal-pairs "$validation_dir/demo_data/manifest.jsonl" --output "$validation_dir/goal_pairs" --pairs-per-episode 2
python -m physical_ai make-fixture --output "$validation_dir/golden_fixture"
python -m physical_ai probe "$validation_dir/golden_fixture" --output "$validation_dir/probe_cpu_a" --learning-steps 6
python -m physical_ai probe "$validation_dir/golden_fixture" --output "$validation_dir/probe_cpu_b" --learning-steps 6
python -m physical_ai compare "$validation_dir/probe_cpu_a" "$validation_dir/probe_cpu_b" --output "$validation_dir/cpu_self_parity.json" --atol 0 --rtol 0
python -m physical_ai budget > "$validation_dir/token_budget.json"
python -m compileall -q src scripts
