#!/usr/bin/env python3
"""Resolve immutable model provenance on a networked host. Config-only by default."""
import argparse
import json
from pathlib import Path
from physical_ai.io import sha256_file, write_json

def main():
    p=argparse.ArgumentParser()
    p.add_argument("repo_id"); p.add_argument("--revision", default="main"); p.add_argument("--output", required=True)
    p.add_argument("--download-weights", action="store_true", help="Explicitly allow a potentially very large download")
    a=p.parse_args()
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except ImportError as exc: raise RuntimeError("Install huggingface-hub in this environment") from exc
    info=HfApi().model_info(a.repo_id, revision=a.revision)
    revision=info.sha
    if not revision: raise RuntimeError("model revision did not resolve")
    out=Path(a.output); out.mkdir(parents=True, exist_ok=True)
    path=hf_hub_download(a.repo_id, "config.json", revision=revision, local_dir=out)
    record={"repo_id":a.repo_id,"revision":revision,"config_sha256":sha256_file(path),"weights_downloaded":False}
    if a.download_weights:
        snapshot_download(a.repo_id, revision=revision, local_dir=out)
        record["weights_downloaded"]=True
    write_json(out/'model_provenance.json',record)
    print(json.dumps(record,indent=2))
if __name__=='__main__': main()
