#!/usr/bin/env python3
"""Exercise vocabulary resizing/resume on a tiny random native Qwen model."""
import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3_5Config, Qwen3_5ForConditionalGeneration

from physical_ai.io import write_json
from physical_ai.vocabulary import expand_action_vocabulary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("new output directory required")
    args.output.mkdir(parents=True)
    reports = []
    for tied in (True, False):
        torch.manual_seed(7)
        config = Qwen3_5Config(text_config={"vocab_size": 8, "hidden_size": 32,
            "intermediate_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 8, "layer_types": ["full_attention"],
            "tie_word_embeddings": tied}, vision_config={"depth": 1, "hidden_size": 32,
            "intermediate_size": 64, "num_heads": 4, "out_hidden_size": 32}, tie_word_embeddings=tied)
        model = Qwen3_5ForConditionalGeneration(config)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(8)}, unk_token="0")))
        old = model.get_input_embeddings().weight.detach().clone()
        result = expand_action_vocabulary(tokenizer, model, count=3)
        torch.testing.assert_close(model.get_input_embeddings().weight[:8], old, rtol=0, atol=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = model.get_input_embeddings()(torch.tensor([8, 9, 10])).square().sum()
        loss.backward()
        if not model.get_input_embeddings().weight.grad[8:].abs().sum() > 0:
            raise ValueError("new action rows did not receive gradients")
        optimizer.step()
        dest = args.output / ("tied" if tied else "untied")
        model.save_pretrained(dest)
        tokenizer.save_pretrained(dest)
        torch.save(optimizer.state_dict(), dest / "optimizer.pt")
        restored = Qwen3_5ForConditionalGeneration.from_pretrained(dest, local_files_only=True)
        restored_tokenizer = PreTrainedTokenizerFast.from_pretrained(dest, local_files_only=True)
        repeat = expand_action_vocabulary(restored_tokenizer, restored, count=3)
        if repeat["added"] != 0 or repeat["tied"] != tied:
            raise ValueError("invalid vocabulary resume")
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        restored_optimizer.load_state_dict(torch.load(dest / "optimizer.pt", weights_only=True))
        reports.append({**result, "resume_exact": True, "optimizer_reloaded": True})
    write_json(args.output / "report.json", {"model": "tiny random native Qwen3_5ForConditionalGeneration",
        "pretrained_weights": False, "cases": reports})
    print(reports)


if __name__ == "__main__":
    main()
