#!/usr/bin/env python3
"""Verify native export hashes and record weight shapes without materializing weights."""
import argparse
from pathlib import Path

from physical_ai.io import write_json
from physical_ai.tau_export import audit_tau_libero


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_tau_libero(args.checkpoint)
    write_json(args.output, report)
    print({key: report[key] for key in ("tensor_count", "stored_elements", "elements_by_dtype")})


if __name__ == "__main__":
    main()
