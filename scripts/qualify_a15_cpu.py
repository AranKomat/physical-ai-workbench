#!/usr/bin/env python3
"""Qualify native A1.5 cache wiring with tiny random Qwen/expert components."""
import argparse
from pathlib import Path
import subprocess
import sys
import importlib.metadata
import inspect

import torch
from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5ForConditionalGeneration

from physical_ai.a15_native import A15NativeCacheAdapter
from physical_ai.io import sha256_file, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("new output directory required")
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != "e6fc904f9edbfb14532e97095fc2372202517f76" or subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain"]):
        raise ValueError("clean pinned A1.5 source required")
    args.output.mkdir(parents=True)
    torch.manual_seed(7)
    tiny = args.output / "tiny-qwen"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", revision="15852e8c16360a2fea060d615a32b45270f8a8fc", local_files_only=True)
    tokenizer.add_special_tokens({"additional_special_tokens": [f"<robot_action_{i}>" for i in range(2048)]})
    config = Qwen3_5Config(text_config={"vocab_size": len(tokenizer), "hidden_size": 32,
        "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 8, "layer_types": ["linear_attention", "full_attention"],
        "linear_key_head_dim": 8, "linear_value_head_dim": 8,
        "linear_num_key_heads": 2, "linear_num_value_heads": 2,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": 1.,
                            "mrope_section": [1, 1, 2], "mrope_interleaved": True}},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4, "out_hidden_size": 32})
    base = Qwen3_5ForConditionalGeneration(config)
    base.save_pretrained(tiny)
    tokenizer.save_pretrained(tiny)
    del base
    sys.path.insert(0, str((args.upstream / "src").resolve()))
    from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import InternVLAA15Config
    from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import InternVLAA15, make_att_2d_masks
    native = InternVLAA15(InternVLAA15Config(vlm_model_name_or_path=str(tiny), dtype="float32", device="cpu",
        action_expert_hidden_size=32, action_expert_intermediate_size=64,
        max_state_dim=4, max_action_dim=4, chunk_size=3, n_action_steps=3,
        action_loss_only=True, num_learnable_tokens=1, tokenize_state=False, compile_model=False))
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    positions = torch.arange(3).reshape(1, 1, 3).expand(3, 1, 3)
    embeddings = native.qwen3_5_with_expert.qwen3_5.get_input_embeddings()(ids)
    attention = native._prepare_attention_masks_4d(make_att_2d_masks(mask, torch.ones_like(ids)))
    _, cache = native.qwen3_5_with_expert.forward(attention_mask=attention, position_ids=positions,
        past_key_values=None, inputs_embeds=[embeddings, None], use_cache=True)
    adapter = A15NativeCacheAdapter(native, horizon=3, action_dim=4)
    state, x, time = torch.zeros(1, 4), torch.randn(1, 3, 4), torch.tensor([.5])
    result = adapter(state=state, prefix_ids=ids, prefix_mask=mask, cache=cache,
        max_prefix_position_ids=positions.max(-1, keepdim=True).values, noisy_actions=x, native_time=time)
    result.square().mean().backward()
    grad = native.action_in_proj.weight.grad
    if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() == 0:
        raise ValueError("native action projection gradient missing")
    write_json(args.output / "report.json", {"source_revision": revision,
        "native_forward_backward_passed": True, "shape": list(result.shape),
        "action_projection_grad_norm": float(grad.norm()), "pretrained_weights": False,
        "architecture": "tiny two-layer mixed linear/full Qwen; not full donor dimensions",
        "transformers_patch_sha256": sha256_file(inspect.getfile(Qwen3_5ForConditionalGeneration)),
        "versions": {k: importlib.metadata.version(k) for k in ("torch", "torchvision", "transformers", "diffusers")},
        "prefix": "three synthetic text token IDs; no cameras or teacher-forced actions"})
    print("native tiny A1.5 prefix-cache forward/backward passed")


if __name__ == "__main__":
    main()
