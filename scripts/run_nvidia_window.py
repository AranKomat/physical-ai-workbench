#!/usr/bin/env python3
"""Execute a prepared, bounded CUDA job batch. No provisioning or scheduling.

Preflight every job and required artifact before consuming the NVIDIA window.
The commands in a plan are trusted code: this runner is not a sandbox.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import torch
from physical_ai.io import sha256_file, write_json


def validate_plan(plan: dict, budget_seconds: float, *, execute: bool) -> None:
    """Validate the complete batch before the first command runs."""
    if not isinstance(plan, dict):
        raise ValueError("plan must be a JSON object")
    if not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError("positive finite total budget required")
    if not isinstance(plan.get("ready_for_cuda", False), bool):
        raise ValueError("ready_for_cuda must be boolean")
    if execute and not plan.get("ready_for_cuda", False):
        raise RuntimeError("plan not marked ready: fill native donor jobs/fixtures before requesting the CUDA window")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("plan must contain at least one job")
    names = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("every job must be an object")
        name, cmd = job.get("name"), job.get("argv")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name in names:
            raise ValueError("job names must be unique, nonempty and safe filename components")
        names.add(name)
        if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) and x and "\x00" not in x for x in cmd):
            raise ValueError("job argv must be a nonempty argument array without NUL characters")
        timeout = job.get("timeout_seconds", budget_seconds)
        if isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f"positive finite timeout required: {name}")
        if job.get("cwd") is not None and not Path(job["cwd"]).is_dir():
            raise ValueError(f"job cwd does not exist: {name}")
    files = plan.get("required_files", [])
    if not isinstance(files, list):
        raise ValueError("required_files must be an array")
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("required artifact needs a path")
        digest = item.get("sha256", "")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("required artifact needs a lowercase SHA-256")
        if not Path(item["path"]).is_file() or sha256_file(item["path"]) != digest:
            raise RuntimeError(f"missing or mismatched required artifact: {item['path']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--budget-seconds", type=float, default=7200)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--output", default="runs/nvidia_session")
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    validate_plan(plan, args.budget_seconds, execute=args.execute)
    if args.execute and not (torch.cuda.is_available() and torch.version.cuda and not torch.version.hip):
        raise RuntimeError("this consolidated reference session requires NVIDIA CUDA; no fallback")
    out = Path(args.output)
    if args.execute and out.exists() and any(out.iterdir()):
        raise RuntimeError("choose an empty output directory; do not overwrite an earlier CUDA session")
    out.mkdir(parents=True, exist_ok=True)
    results = []
    start = time.monotonic()
    try:
        for job in plan["jobs"]:
            cmd = job["argv"]
            if not args.execute:
                results.append({"name": job["name"], "argv": cmd, "status": "dry_run"})
                continue
            remaining = args.budget_seconds - (time.monotonic() - start)
            if remaining <= 0:
                results.append({"name": job["name"], "status": "budget_exhausted"})
                break
            limit = min(remaining, float(job.get("timeout_seconds", remaining)))
            logfile = out / f"{job['name']}.log"
            with logfile.open("w") as log:
                proc = subprocess.Popen(cmd, cwd=job.get("cwd"), stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = proc.wait(timeout=limit)
                    status = "passed" if code == 0 else "failed"
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    code, status = -9, "timed_out"
                except BaseException:
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    raise
            results.append({"name": job["name"], "returncode": code, "status": status, "log": str(logfile)})
            write_json(out / "session.json", {"jobs": results, "elapsed_seconds": time.monotonic() - start,
                                              "executed": args.execute})
            if status != "passed":
                break
    finally:
        write_json(out / "session.json", {"jobs": results, "elapsed_seconds": time.monotonic() - start,
                   "executed": args.execute, "budget_seconds": args.budget_seconds,
                   "all_planned_jobs_recorded": len(results) == len(plan["jobs"]),
                   "native_vla_qualified": False,
                   "note": "Job exit codes alone are not model qualification; review the numeric/closed-loop reports."})
    return int(len(results) != len(plan["jobs"]) or any(j["status"] not in {"passed", "dry_run"} for j in results))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
