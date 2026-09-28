import pytest
import torch
from PIL import Image

from physical_ai.collation import ObservationExample, QwenObservationCollator


class Processor:
    """Char tokenizer makes the exact target boundary independently inspectable."""
    def __init__(self, left=False):
        self.left = left

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        prompt = "USER:" + messages[0]["content"][-1]["text"] + "\nASSISTANT:"
        if add_generation_prompt:
            return prompt
        return prompt + messages[1]["content"][0]["text"] + "!"

    def __call__(self, text, images, padding, return_tensors):
        width = max(map(len, text))
        ids = torch.zeros(len(text), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, value in enumerate(text):
            start = width - len(value) if self.left else 0
            ids[i, start:start + len(value)] = torch.tensor(list(map(ord, value)))
            mask[i, start:start + len(value)] = 1
        return {"input_ids": ids, "attention_mask": mask,
                "pixel_values": torch.ones(len(text), 3, 2, 2)}


def examples(answer="XYZ"):
    image = Image.new("RGB", (32, 32))
    return [ObservationExample("lift", (image,), answer, "test-labels:v1"),
            ObservationExample("lower the object", (image,), "Q", "test-labels:v1")]


@pytest.mark.parametrize("left", [False, True])
def test_only_answer_and_terminator_are_supervised(left):
    batch = QwenObservationCollator(Processor(left))(examples())
    aux = batch["auxiliary_inputs"]
    for row, expected in enumerate(["XYZ!", "Q!"]):
        actual = aux["labels"][row]
        assert actual[actual != -100].tolist() == list(map(ord, expected))
    assert (aux["labels"][aux["attention_mask"] == 0] == -100).all()
    assert "labels" not in batch["prefix_inputs"]


def test_changing_target_cannot_change_motor_input():
    collate = QwenObservationCollator(Processor())
    a, b = collate(examples()), collate(examples("different answer"))
    for key in a["prefix_inputs"]:
        assert torch.equal(a["prefix_inputs"][key], b["prefix_inputs"][key])


def test_unstable_template_fails():
    class Unstable(Processor):
        def apply_chat_template(self, *args, **kwargs):
            text = super().apply_chat_template(*args, **kwargs)
            return text if kwargs["add_generation_prompt"] else "OTHER:" + text
    with pytest.raises(ValueError, match="template changed"):
        QwenObservationCollator(Unstable())(examples())


def test_source_required():
    ex = examples()[0]
    with pytest.raises(ValueError, match="label source"):
        QwenObservationCollator(Processor())([
            ObservationExample(ex.instruction, ex.images, ex.auxiliary_answer, "")
        ])
