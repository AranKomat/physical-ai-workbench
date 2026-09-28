#!/usr/bin/env python3
"""Local Qwen multimodal forward/backward probe. No downloads; no robot performance claim."""
import argparse
import json
from pathlib import Path
import time
import torch
from PIL import Image
from physical_ai.native import HuggingFaceQwen
from physical_ai.io import write_json
from physical_ai.validation import require_device, hardware_info

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model', required=True); p.add_argument('--image', required=True)
    p.add_argument('--output', required=True); p.add_argument('--device', default='cuda')
    p.add_argument('--precision', choices=['bf16','fp32'], default='bf16')
    p.add_argument('--prompt', default='Describe the target object and a safe next subtask.')
    p.add_argument('--backward', action='store_true')
    a=p.parse_args(); require_device(a.device)
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
    image=Image.open(a.image).convert('RGB')
    messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':a.prompt}]}]
    inputs=processor.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors='pt',add_generation_prompt=True)
    inputs={k:v.to(a.device) if torch.is_tensor(v) else v for k,v in inputs.items()}
    dtype=torch.bfloat16 if a.precision=='bf16' else torch.float32
    brain=HuggingFaceQwen.from_pretrained(a.model,revision=None,dtype=dtype,local_files_only=True).to(a.device)
    brain.train(a.backward)
    start=time.perf_counter()
    with torch.set_grad_enabled(a.backward):
        layers,mask=brain.prefix(inputs)
        # A numeric gradient probe only, not a training objective for the real model.
        probe_loss=layers[-1].float().square().mean()
        if a.backward: probe_loss.backward()
    if a.device.startswith('cuda'): torch.cuda.synchronize()
    report={'hardware':hardware_info(),'model':a.model,'device':a.device,'precision':a.precision,
            'hidden_shape':list(layers[-1].shape),'input_tokens':int(mask.sum()),'probe_loss':float(probe_loss.detach()),
            'backward':a.backward,'seconds':time.perf_counter()-start,'finite':bool(torch.isfinite(layers[-1]).all()),
            'native_VLA_validated':False,'note':'Single VLM numeric probe only; does not validate action semantics or Primus.'}
    write_json(a.output,report); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
