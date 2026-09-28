#!/usr/bin/env python3
"""Real W0 source, reduced random architecture, CPU gradient/adapter checks."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from physical_ai.io import sha256_file, write_json
from physical_ai.native import NativeFlowConvention, W0MotorAdapter, load_w0_motor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != "90801baa3bdc7829c1e4989edfe3d0fb2410ea6a" or subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain"]):
        raise ValueError("clean pinned W0 source required")
    sys.path.insert(0, str((args.upstream / "src").resolve()))
    from wam.model.modules.experts.action_dit import ActionDiT
    torch.manual_seed(7)
    constructor = dict(hidden_dim=32, action_dim=4, ffn_dim=64, text_dim=16, freq_dim=8,
                       eps=1e-6, num_heads=4, attn_head_dim=8, num_layers=2)
    native = ActionDiT(**constructor)
    native.configure_vlm_conditioning(12)
    convention = NativeFlowConvention(1000, "data_to_noise", -1)
    adapter = W0MotorAdapter(native, 12, 4, convention)
    x, t = torch.randn(2, 3, 4), torch.tensor([.2, .7])
    context = torch.randn(2, 5, 12)
    cmask = torch.ones(2, 5, dtype=torch.bool)
    state = torch.randn(2, 4)
    smask = torch.ones(2, 4, dtype=torch.bool)
    amask = torch.ones(2, 3, 4, dtype=torch.bool)
    actual = adapter(x, t, context, cmask, state, smask, amask)
    st = adapter.state_adapter(torch.cat((state, smask.float()), -1))[:, None]
    expected = -native(action_tokens=x, timestep=(1-t)*1000,
        context=torch.cat((context, st), 1), context_mask=torch.ones(2, 6, dtype=torch.bool))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    grads = {k: float(v.grad.norm()) for k, v in adapter.named_parameters() if v.grad is not None}
    if not grads or not all(torch.isfinite(torch.tensor(v)) for v in grads.values()):
        raise ValueError("invalid native gradient")
    if grads.get("state_adapter.weight", 0) <= 0:
        raise ValueError("state conditioning did not receive gradient")
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "tiny-native.pt"
        torch.save({"mot": {"mixtures.action." + k: v for k, v in native.state_dict().items()}}, checkpoint)
        restored, load_report = load_w0_motor(checkpoint=str(checkpoint), constructor_config=constructor,
            state_dict_key="mot", source_prefix="mixtures.action.", context_dim=12,
            convention=convention, device="cpu", dtype=torch.float32, reset_action_io=False,
            checkpoint_sha256=sha256_file(checkpoint), qualification_mode=True, source_context_dim=12)
        for key, tensor in native.state_dict().items():
            torch.testing.assert_close(tensor, restored.action_dit.state_dict()[key], rtol=0, atol=0)
        if load_report["new_cross_attention_blocks"] != 0:
            raise ValueError("trained cross-attention was reset")
    write_json(args.output, {"source_revision": revision, "seed": 7, "shape": list(actual.shape),
        "native_wrapper_exact": True, "gradient_norms": grads, "pretrained_weights": False,
        "preconditioned_checkpoint_roundtrip_exact": True, "load_report": load_report,
        "note": "Native reduced random model; no CUDA/ROCm or pretrained policy equivalence claim."})
    print({"native_wrapper_exact": True, "gradient_tensors": len(grads)})


if __name__ == "__main__":
    main()
