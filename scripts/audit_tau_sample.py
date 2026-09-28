#!/usr/bin/env python3
"""Audit bundled A2D records without remapping them into LIBERO or research slots."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from physical_ai.io import sha256_file, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    info = json.loads((args.root / "meta/info.json").read_text())
    files = sorted((args.root / "data").rglob("*.parquet"))
    if not files:
        raise ValueError("no real parquet records found")
    rows = [row for file in files for row in pq.read_table(file).to_pylist()]
    if len(rows) != info["total_frames"]:
        raise ValueError("frame count differs from metadata")
    episodes = {}
    for row in rows:
        episodes.setdefault(row["episode_index"], []).append(row)
        for key in ("observation.state", "action"):
            value = np.asarray(row[key])
            if list(value.shape) != info["features"][key]["shape"] or not np.isfinite(value).all():
                raise ValueError(f"invalid {key}")
    if len(episodes) != info["total_episodes"]:
        raise ValueError("episode count differs from metadata")
    lengths = {}
    for episode, records in episodes.items():
        frame = np.asarray([r["frame_index"] for r in records])
        stamp = np.asarray([r["timestamp"] for r in records])
        if not np.array_equal(frame, np.arange(len(records))):
            raise ValueError(f"noncontiguous episode {episode}")
        if not np.isfinite(stamp).all() or not np.allclose(np.diff(stamp), 1 / info["fps"], atol=1e-4):
            raise ValueError(f"invalid timestamps in episode {episode}")
        lengths[str(episode)] = len(records)
    videos = sorted((args.root / "videos").rglob("*.mp4"))
    report = {
        "robot_type": info["robot_type"], "frames": len(rows),
        "episode_lengths": lengths, "fps": info["fps"],
        "state_shape": info["features"]["observation.state"]["shape"],
        "action_shape": info["features"]["action"]["shape"],
        "complete_10_frame_windows": sum(max(0, n - 9) for n in lengths.values()),
        "files": {str(f.relative_to(args.root)): sha256_file(f)
                  for f in [args.root / "meta/info.json", *files, *videos]},
        "video_decode_verified": False,
        "native_conversion_verified": False,
        "compatible_with_libero_checkpoint": False,
        "note": "Real A2D data audit only; windows are not yet encoded training examples.",
    }
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
