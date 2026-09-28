#!/usr/bin/env python3
"""Export a bounded, episode-spread native/FAST/Qwen CPU qualification subset."""
import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch
from transformers import AutoProcessor

from physical_ai.action_labels import checked_fast_labels
from physical_ai.io import sha256_file, write_json
from physical_ai.native_batches import NativeWindow, NativeWindowCollator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--fast", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("new output directory required")
    # Reuse the clean-source and reviewed-code contracts from the qualification.
    import subprocess
    expected = "f1665fbaf624d1468b168e3a54cccc9e326212d9"
    if subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip() != expected:
        raise ValueError("wrong native source revision")
    if subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain"]):
        raise ValueError("dirty native source")
    if sha256_file(args.fast / "processing_action_tokenizer.py") != "6f021ca1f4c1b194ab6fa399d80baf3d642eadb17efb8f73301e4ac401522c20":
        raise ValueError("unreviewed codec source")
    fast = AutoProcessor.from_pretrained(args.fast, trust_remote_code=True, local_files_only=True,
                                        tokenizer_file=str(args.fast / "tokenizer.json"))
    qwen = AutoProcessor.from_pretrained("Qwen/Qwen3.5-2B", revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
                                         local_files_only=True)
    names = [f"<robot_action_{i}>" for i in range(2048)]
    qwen.tokenizer.add_special_tokens({"additional_special_tokens": names})
    if qwen.tokenizer.convert_tokens_to_ids(names) != list(range(248077, 250125)):
        raise ValueError("wrong action vocabulary")
    sys.path[:0] = [str(args.upstream.resolve()), str((args.upstream / "src").resolve())]
    from configs.example_agibot_world_gong.data import agibot_world_gong_ft
    from tau0_vla.data import FinchDataLoader
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    loader = FinchDataLoader(agibot_world_gong_ft(), batch_size=1, num_workers=0, video_backend="pyav")
    node = loader.dataset
    while type(node).__name__ != "_RangeFilteredDataset":
        node = node._dataset
    table = pq.read_table(args.upstream / "example_data/data/chunk-000/file-000.parquet",
                          columns=["episode_index", "frame_index", "timestamp"]).to_pydict()
    episode = np.asarray(table["episode_index"])
    frame = np.asarray(table["frame_index"])
    times = np.asarray(table["timestamp"])
    normalization = sha256_file(args.upstream / "configs/example_agibot_world_gong/norm_stats.json")
    windows, records = [], []
    offset = 0
    for start, end in node._ranges:
        start, end = int(start), int(end)
        if end - start >= 30:
            for raw in sorted({start, start + (end - start - 30) // 2}):
                if len(records) >= 50:
                    break
                filtered = offset + raw - start
                sample = loader.dataset[filtered]
                actions = sample["extras"]["action"]["q_norm"]
                if actions is None or not np.all(episode[raw:raw + 30] == episode[raw]):
                    raise ValueError("invalid normalized or cross-episode target")
                if not np.array_equal(frame[raw:raw + 30], np.arange(frame[raw], frame[raw] + 30)):
                    raise ValueError("noncontiguous source frames")
                labels = checked_fast_labels(fast, actions)
                active = np.asarray(sample["action_mask"], dtype=bool)
                record = {"episode": int(episode[raw]), "frame": int(frame[raw]), "filtered_index": filtered,
                          "tokens": len(labels["codes"]), "max_error": labels["max_absolute_error"],
                          "active_rmse": float(np.sqrt(np.mean(np.asarray(labels["per_dimension_rmse"])[active] ** 2)))}
                records.append(record)
                cameras = ("head", "wrist_left", "wrist_right")
                windows.append(NativeWindow(expected, str(episode[raw]), "tau-gong-joint40-h30",
                    "action=q_norm:" + normalization + ";state=native", int(frame[raw]), int(np.sum(episode == episode[raw])),
                    times[raw:raw + 30], cameras, (float(times[raw]),) * 3,
                    tuple(Image.fromarray(sample["images"][k]) for k in cameras), sample["prompt"],
                    labels["text"], "fast:ec4d7aa71691cac0b8bed6942be45684db2110f4", sample["state"],
                    np.asarray(sample["state_mask"], bool), actions,
                    np.broadcast_to(active, actions.shape).copy()))
        offset += end - start
    args.output.mkdir(parents=True)
    collate = NativeWindowCollator(qwen)
    batches = []
    for start in range(0, len(windows), 4):
        subset = windows[start:start + 4]
        batch = collate(subset)
        for row, window in enumerate(subset):
            expected_codes = qwen.tokenizer.encode(window.auxiliary_answer, add_special_tokens=False)
            actual = batch["auxiliary_inputs"]["labels"][row]
            actual = actual[(actual >= 248077) & (actual <= 250124)].tolist()
            if actual != expected_codes:
                raise ValueError("batch truncated or changed FAST labels")
        path = args.output / f"batch-{start // 4:03d}.pt"
        torch.save(batch, path)
        torch.load(path, weights_only=True)
        batches.append({"path": path.name, "sha256": sha256_file(path), "windows": len(subset)})
    qwen.save_pretrained(args.output / "processor")
    write_json(args.output / "manifest.json", {
        "windows": len(windows), "episodes": sorted({r["episode"] for r in records}),
        "records": records, "batches": batches, "normalization_sha256": normalization,
        "camera_timestamp_semantics": "native LeRobot requested timestamps; decoding uses its tolerance checks",
        "native_training_augmentation": True, "seed": 7,
        "purpose": "integration subset, not a held-out benchmark or production training dataset",
        "pretrained_model_executed": False,
    })
    print(json.dumps({"windows": len(windows), "episodes": len({r["episode"] for r in records}), "batches": len(batches)}))


if __name__ == "__main__":
    main()
