"""Native processor batches with independently rendered motor and CE inputs.

Auxiliary answers are supplied by a versioned external codec/label pipeline.
This module does not invent an action tokenizer or accept future goal images.
"""
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image


@dataclass(frozen=True)
class ObservationExample:
    instruction: str
    images: tuple[Image.Image, ...]
    auxiliary_answer: str
    auxiliary_source: str


class QwenObservationCollator:
    """Preserve processor image expansion and mask all prompt/padding CE tokens.

    Only current/legally retained observations belong in ``images``. Callers
    own camera ordering, timestamps and provenance. Auxiliary labels must be
    produced upstream using a named codec or annotation revision.
    """

    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(self, examples: list[ObservationExample]) -> dict:
        if not examples:
            raise ValueError("at least one observation is required")
        prefix_texts, full_texts, image_batches = [], [], []
        for example in examples:
            if not example.instruction.strip() or not example.auxiliary_answer.strip():
                raise ValueError("nonempty instruction and auxiliary answer required")
            if not example.auxiliary_source.strip():
                raise ValueError("versioned auxiliary label source required")
            if not example.images or any(not isinstance(im, Image.Image) for im in example.images):
                raise ValueError("observation images must be nonempty PIL image tuples")
            content = [{"type": "image"} for _ in example.images]
            content.append({"type": "text", "text": example.instruction})
            messages = [{"role": "user", "content": content}]
            # Render twice: motor context never receives the auxiliary answer.
            prefix_texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            ))
            full_texts.append(self.processor.apply_chat_template(
                messages + [{"role": "assistant", "content": [
                    {"type": "text", "text": example.auxiliary_answer}
                ]}], tokenize=False, add_generation_prompt=False,
            ))
            image_batches.append([im.convert("RGB") for im in example.images])

        prefix = dict(self.processor(text=prefix_texts, images=image_batches,
                                     padding=True, return_tensors="pt"))
        auxiliary = dict(self.processor(text=full_texts, images=image_batches,
                                        padding=True, return_tensors="pt"))
        labels = torch.full_like(auxiliary["input_ids"], -100)
        counts = []
        for row in range(len(examples)):
            prefix_ids = prefix["input_ids"][row][prefix["attention_mask"][row].bool()]
            positions = auxiliary["attention_mask"][row].bool().nonzero().flatten()
            full_ids = auxiliary["input_ids"][row, positions]
            n = len(prefix_ids)
            if len(full_ids) <= n or not torch.equal(prefix_ids, full_ids[:n]):
                raise ValueError("assistant template changed the prefix; explicit template adapter required")
            labels[row, positions[n:]] = full_ids[n:]
            counts.append(len(positions) - n)

        # The two forwards must see identical observation pixels and geometry.
        for key in ("pixel_values", "image_grid_thw"):
            if key in prefix and (key not in auxiliary or not torch.equal(prefix[key], auxiliary[key])):
                raise ValueError(f"processor changed observation field between forwards: {key}")
        auxiliary["labels"] = labels
        return {
            "prefix_inputs": prefix,
            "auxiliary_inputs": auxiliary,
            "collation_metadata": {
                "auxiliary_sources": [ex.auxiliary_source for ex in examples],
                "prefix_tokens": prefix["attention_mask"].sum(-1).tolist(),
                "supervised_tokens": counts,
            },
        }
