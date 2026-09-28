#!/usr/bin/env python3
"""Real-data FAST and Qwen collation probe; no model weights or training."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor

from physical_ai.action_labels import checked_fast_labels
from physical_ai.collation import ObservationExample, QwenObservationCollator
from physical_ai.io import sha256_file, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--fast", type=Path, required=True)
    p.add_argument("--qwen", default="Qwen/Qwen3.5-2B")
    p.add_argument("--qwen-revision", default="15852e8c16360a2fea060d615a32b45270f8a8fc")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "upstreams.lock.json").read_text())
    expected = next(r["revision"] for r in lock["repositories"] if r["name"] == "tau0")
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != expected or subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain"]):
        raise ValueError("clean pinned Tau checkout required")
    if sha256_file(args.fast / "processing_action_tokenizer.py") != "6f021ca1f4c1b194ab6fa399d80baf3d642eadb17efb8f73301e4ac401522c20":
        raise ValueError("FAST implementation differs from reviewed source")
    # A1.5 explicitly supplies tokenizer_file for FAST under Transformers 5.
    fast = AutoProcessor.from_pretrained(args.fast, trust_remote_code=True, local_files_only=True,
                                        tokenizer_file=str(args.fast / "tokenizer.json"))
    processor = AutoProcessor.from_pretrained(args.qwen, revision=args.qwen_revision,
                                             trust_remote_code=False, local_files_only=True)
    tokens = [f"<robot_action_{i}>" for i in range(2048)]
    vocab = processor.tokenizer.get_vocab()
    present = [t in vocab for t in tokens]
    if any(present) and not all(present):
        raise ValueError("partial action-token vocabulary")
    if not any(present):
        processor.tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    if processor.tokenizer.convert_tokens_to_ids(tokens) != list(range(248077, 250125)):
        raise ValueError("tokenizer does not have the audited A1.5-style action-token mapping")
    sys.path[:0] = [str(args.upstream.resolve()), str((args.upstream / "src").resolve())]
    from configs.example_agibot_world_gong.data import agibot_world_gong_ft
    from tau0_vla.data import FinchDataLoader

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    loader = FinchDataLoader(agibot_world_gong_ft(), batch_size=1, num_workers=0, video_backend="pyav")
    examples, records = [], []
    for index in (0, len(loader.dataset) // 2, len(loader.dataset) - 1):
        sample = loader.dataset[index]
        actions = sample["extras"]["action"]["q_norm"]
        if actions is None:
            raise ValueError("native quantile normalization unavailable; no raw-action fallback")
        labels = checked_fast_labels(fast, actions)
        active = np.asarray(sample["action_mask"]).astype(bool)
        labels["active_dimensions"] = np.flatnonzero(active).tolist()
        labels["active_rmse"] = float(np.sqrt(np.mean(np.asarray(labels["per_dimension_rmse"])[active] ** 2)))
        images = tuple(Image.fromarray(sample["images"][key]) for key in ("head", "wrist_left", "wrist_right"))
        examples.append(ObservationExample(sample["prompt"], images, labels["text"],
                                          "fast:ec4d7aa71691cac0b8bed6942be45684db2110f4;native-gong-q_norm"))
        records.append({"filtered_index": index, **{k: v for k, v in labels.items() if k != "text"}})
    collate = QwenObservationCollator(processor)
    batch = collate(examples)
    changed = collate([ObservationExample(e.instruction, e.images, "<robot_action_0>", e.auxiliary_source) for e in examples])
    for key, value in batch["prefix_inputs"].items():
        if not torch.equal(value, changed["prefix_inputs"][key]):
            raise ValueError(f"target leakage in {key}")
    for row, record in enumerate(records):
        labels = batch["auxiliary_inputs"]["labels"][row]
        actual = labels[(labels >= 248077) & (labels <= 250124)].tolist()
        if actual != [248077 + i for i in record["codes"]]:
            raise ValueError("collation changed or truncated action labels")
    write_json(args.output, {
        "tau_revision": revision, "fast_revision": "ec4d7aa71691cac0b8bed6942be45684db2110f4",
        "qwen_model": args.qwen, "qwen_revision": args.qwen_revision,
        "action_token_ids": [248077, 250124], "model_embeddings_resized": False,
        "fast_files": {f.name: sha256_file(f) for f in args.fast.iterdir() if f.is_file()},
        "normalization_sha256": sha256_file(args.upstream / "configs/example_agibot_world_gong/norm_stats.json"),
        "samples": records, "collation": batch["collation_metadata"],
        "target_independence_passed": True, "action_token_roundtrip_passed": True,
        "representation": "native Tau 30x40 quantile-normalized; not A1.5 50x32 or research 80D",
        "pretrained_forward_or_training_executed": False,
    })
    print(json.dumps(batch["collation_metadata"], indent=2))


if __name__ == "__main__":
    main()
