#!/usr/bin/env python3
"""Clone pinned upstreams on a networked host. Dry-run by default; no pip installs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


def commands(entry, root):
    revision = entry.get("revision")
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{entry['name']}: unresolved revision; resolve and review before cloning")
    dst = root / entry["name"]
    return dst, [["git", "init", str(dst)], ["git", "-C", str(dst), "remote", "add", "origin", entry["url"]],
                 ["git", "-C", str(dst), "fetch", "--depth", "1", "origin", revision],
                 ["git", "-C", str(dst), "checkout", "--detach", "FETCH_HEAD"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lock", default="upstreams.lock.json"); p.add_argument("--root", default="third_party")
    p.add_argument("--names", nargs="*"); p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    entries = json.loads(Path(args.lock).read_text())["repositories"]
    chosen = set(args.names) if args.names else {e["name"] for e in entries if e.get("core")}
    if chosen - {e["name"] for e in entries}: p.error("unknown upstream name")
    root = Path(args.root).resolve()
    for e in entries:
        if e["name"] not in chosen: continue
        dst, cmds = commands(e, root)
        if dst.exists():
            r = subprocess.run(["git", "-C", str(dst), "rev-parse", "HEAD"], capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip() == e["revision"]:
                print(f"already pinned: {dst}"); continue
            raise RuntimeError(f"existing checkout not pinned: {dst}; preserve your changes and reconcile manually")
        for cmd in cmds:
            print(json.dumps(cmd))
            if args.execute:
                root.mkdir(parents=True, exist_ok=True)
                subprocess.run(cmd, check=True, timeout=300)
        if args.execute:
            got = subprocess.check_output(["git", "-C", str(dst), "rev-parse", "HEAD"], text=True).strip()
            if got != e["revision"]: raise RuntimeError("unexpected fetched revision")
    return 0

if __name__ == "__main__":
    try: raise SystemExit(main())
    except (ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr); raise SystemExit(2)
