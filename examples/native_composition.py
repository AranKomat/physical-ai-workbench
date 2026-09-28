"""Minimal native composition skeleton using implemented adapters.

Not executable without real local weights, audited checkpoint keys, qualified
flow convention, upstream W0 environment, HF processor, and a real batch collator.
This example does not download anything or infer the missing schema.
"""
import json
from pathlib import Path
import torch
from physical_ai.native import (HuggingFaceQwen, NativeFlowConvention,
                                load_w0_motor, NativeQwenFlowPolicy)


def build(local_qwen: str, w0_manifest: str, device: str = "cuda"):
    record = json.loads(Path(w0_manifest).read_text())
    # The boolean must be backed by a saved qualification report on your host.
    convention = NativeFlowConvention(**record["flow_convention"])
    brain = HuggingFaceQwen.from_pretrained(local_qwen, revision=None, dtype=torch.bfloat16,
                                           selected_layers=(-9, -5, -1), local_files_only=True).to(device)
    motor, audit = load_w0_motor(
        checkpoint=record["checkpoint"], checkpoint_sha256=record["sha256"],
        constructor_config=record["constructor_config"], state_dict_key=record["state_dict_key"],
        source_prefix=record["action_prefix"], context_dim=brain.hidden_size,
        convention=convention, device=device, dtype=torch.bfloat16,
        reset_action_io=True, qualification_mode=False)
    policy = NativeQwenFlowPolicy(brain, motor, insulated=True).to(device=device, dtype=torch.bfloat16)
    return policy, audit

# batch = {
#   "prefix_inputs": native_processor(current_images_and_available_context),
#   "auxiliary_inputs": assistant_only_CE_inputs_with_validated_action_tokens,
#   "state": ..., "state_mask": ..., "actions": ..., "action_mask": ...
# }
# Never give prefix_inputs the teacher-forced action answer.
# losses = policy(batch, fixed_noise, sampled_time)
# losses["total"].backward()
