#!/usr/bin/env python3
"""Download processor files only and exercise native multimodal collation.

No pretrained model weights are downloaded or executed. Synthetic observations
and text labels qualify preprocessing, not a policy or action tokenizer.
"""
import argparse
import json
import platform
from pathlib import Path

import torch
import transformers
from huggingface_hub import HfApi, snapshot_download
from PIL import Image
from transformers import AutoProcessor

from physical_ai.collation import ObservationExample, QwenObservationCollator
from physical_ai.io import sha256_file, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    revision = HfApi().model_info(args.model, revision=args.revision).sha
    path = snapshot_download(args.model, revision=revision, allow_patterns=[
        "*.json", "*.jinja", "*.txt", "*.model",
    ])
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=False, local_files_only=True)
    examples = [ObservationExample("Lift the object.", (Image.new("RGB", (224, 224)),),
                                    "lift", "synthetic-text:v1"),
                ObservationExample("Move the object beside the box.",
                                   (Image.new("RGB", (224, 224)), Image.new("RGB", (224, 224))),
                                   "move beside box", "synthetic-text:v1")]
    collate = QwenObservationCollator(processor)
    batch = collate(examples)
    changed = collate([ObservationExample(ex.instruction, ex.images, "different target", ex.auxiliary_source)
                       for ex in examples])
    for key in batch["prefix_inputs"]:
        if not torch.equal(batch["prefix_inputs"][key], changed["prefix_inputs"][key]):
            raise RuntimeError(f"auxiliary target leaked into motor prefix: {key}")
    report = {
        "model": args.model, "revision": revision,
        "processor_class": type(processor).__name__,
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__,
        "processor_file_hashes": {f.name: sha256_file(f) for f in Path(path).iterdir()
                                  if f.is_file() and f.suffix in {".json", ".jinja", ".txt", ".model"}},
        "prefix_shapes": {k: list(v.shape) for k, v in batch["prefix_inputs"].items()},
        "metadata": batch["collation_metadata"],
        "target_independence_passed": True, "pretrained_weights_executed": False,
        "real_robot_data": False,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
