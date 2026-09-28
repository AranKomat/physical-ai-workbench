from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from .io import read_jsonl, write_json, resolve_local


def parser():
    p = argparse.ArgumentParser(description="Physical-AI research workbench; CPU-tested components, not a trained policy")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("smoke", help="Train a tiny synthetic CPU reference; no model downloads")
    s.add_argument("--output", default="runs/cpu_smoke"); s.add_argument("--steps", type=int, default=40)
    s = sub.add_parser("make-demo-data", help="Create explicitly synthetic episode fixtures")
    s.add_argument("--output", default="runs/demo_data"); s.add_argument("--episodes", type=int, default=6)
    s = sub.add_parser("audit-data", help="Audit the local episode interchange, not raw vendor datasets")
    s.add_argument("manifest"); s.add_argument("--output", default="runs/data_audit.json")
    s = sub.add_parser("goal-pairs", help="Export real future-frame pairs; generation is NOT performed")
    s.add_argument("manifest"); s.add_argument("--output", default="runs/goal_pairs")
    s.add_argument("--pairs-per-episode", type=int, default=8)
    s = sub.add_parser("budget", help="Token/hour estimate; explicitly not measured throughput")
    s.add_argument("--width", type=int, default=256); s.add_argument("--height", type=int, default=256)
    s.add_argument("--images", type=int, default=9); s.add_argument("--text-tokens", type=int, default=150)
    s.add_argument("--anchor-hz", type=float, default=1); s.add_argument("--tokens-day", type=float, default=2e9)
    s.add_argument("--robot-token-fraction", type=float, default=.9)
    s = sub.add_parser("make-fixture", help="Create portable random reference-model oracle tensors")
    s.add_argument("--output", default="runs/golden_fixture"); s.add_argument("--seed", type=int, default=17)
    s = sub.add_parser("probe", help="Run reference fixture; no native/GPU qualification claim")
    s.add_argument("fixture"); s.add_argument("--output", required=True)
    s.add_argument("--device", default="cpu"); s.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    s.add_argument("--learning-steps", type=int, default=12)
    s = sub.add_parser("compare", help="Compare saved probe tensors with explicit tolerances")
    s.add_argument("reference"); s.add_argument("candidate"); s.add_argument("--output", default="runs/parity.json")
    s.add_argument("--atol", type=float, default=1e-5); s.add_argument("--rtol", type=float, default=1e-4)
    s = sub.add_parser("inspect-checkpoint", help="Inspect local tensor keys and parameter counts safely")
    s.add_argument("path"); s.add_argument("--key-path", default=""); s.add_argument("--prefix", default="")
    s.add_argument("--output", default="runs/checkpoint_audit.json")
    s = sub.add_parser("inspect-hf-config", help="Read a LOCAL HF config; never downloads weights")
    s.add_argument("path")
    sub.add_parser("doctor", help="Show installed runtime and GPU availability")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "smoke":
            from .training import run_cpu_smoke
            result = run_cpu_smoke(args.output, args.steps)
        elif args.command == "make-demo-data":
            from .examples import write_demo_dataset
            result = {"manifest": str(write_demo_dataset(args.output, args.episodes)), "synthetic": True}
        elif args.command in {"audit-data", "goal-pairs"}:
            from .data import load_episode, validate_disjoint_splits, quality_report, goal_pair_manifest
            from .planning import subtask_duration_summary
            root = Path(args.manifest).parent
            rows = list(read_jsonl(args.manifest)); validate_disjoint_splits(rows)
            episodes = [(load_episode(resolve_local(root, row["episode"])), row["split"]) for row in rows]
            for row, (ep, _) in zip(rows, episodes):
                if ep.episode_id != row["episode_id"] or ep.split_group != row["split_group"]:
                    raise ValueError("manifest/episode identity mismatch")
            if args.command == "audit-data":
                result = {"episodes": [quality_report(ep) for ep, _ in episodes],
                          "duration_statistics": subtask_duration_summary([s for ep, _ in episodes for s in ep.segments]),
                          "cross_split_leakage": False, "semantic_quality_verified": False}
                write_json(args.output, result)
            else:
                pairs = goal_pair_manifest(episodes, Path(args.output), args.pairs_per_episode)
                result = {"pairs": len(pairs), "manifest": str(Path(args.output) / "pairs.jsonl"),
                          "generated_images": 0, "warning": "Targets are privileged real-future images, never deployment inputs."}
        elif args.command == "budget":
            from .budget import TokenBudget
            result = TokenBudget(width=args.width, height=args.height, image_count=args.images, text_tokens=args.text_tokens,
                      anchor_hz=args.anchor_hz, measured_total_tokens_per_day=args.tokens_day, robot_token_fraction=args.robot_token_fraction).estimate()
        elif args.command == "make-fixture":
            from .validation import make_fixture
            result = make_fixture(args.output, args.seed)
        elif args.command == "probe":
            from .validation import run_probe
            result = run_probe(args.fixture, args.output, device=args.device, precision=args.precision, learning_steps=args.learning_steps)
        elif args.command == "compare":
            from .validation import compare_probes
            result = compare_probes(args.reference, args.candidate, args.output, atol=args.atol, rtol=args.rtol)
            print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))
            return 0 if result["numeric_pass"] else 2
        elif args.command == "inspect-checkpoint":
            from .checkpoints import read_tensor_state, select_prefix
            from .io import sha256_file
            state = select_prefix(read_tensor_state(args.path, args.key_path), args.prefix)
            result = {"sha256": sha256_file(args.path), "key_count": len(state),
                      "numel": sum(v.numel() for v in state.values()),
                      "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in state.items()}}
            write_json(args.output, result)
            result = {k: v for k, v in result.items() if k != "tensors"}
        elif args.command == "inspect-hf-config":
            cfg = json.loads(Path(args.path).read_text()); text = cfg.get("text_config", cfg)
            result = {"architectures": cfg.get("architectures"), "model_type": cfg.get("model_type"),
                      "text": {k: text.get(k) for k in ("model_type", "hidden_size", "num_hidden_layers", "layer_types", "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size", "num_experts", "num_experts_per_tok")},
                      "vision": cfg.get("vision_config"), "warning": "Same HF model_type does NOT establish checkpoint conversion or Primus kernel parity."}
        else:
            from .validation import hardware_info
            result = hardware_info()
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
