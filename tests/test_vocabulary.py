import io

import pytest
import torch
from torch import nn

from physical_ai.vocabulary import expand_action_vocabulary


class Tokenizer:
    def __init__(self):
        self.vocab = {str(i): i for i in range(8)}

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab.copy()

    def convert_tokens_to_ids(self, names):
        return [self.vocab.get(x) for x in names]

    def add_special_tokens(self, value):
        for name in value["additional_special_tokens"]:
            self.vocab[name] = len(self.vocab)


class Model(nn.Module):
    def __init__(self, tied, rows=8):
        super().__init__()
        self.embed = nn.Embedding(rows, 4)
        self.head = nn.Linear(4, rows, bias=False)
        self.tied = tied
        if tied:
            self.head.weight = self.embed.weight

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head

    def resize_token_embeddings(self, rows):
        old_e, old_h = self.embed.weight.detach().clone(), self.head.weight.detach().clone()
        self.embed = nn.Embedding(rows, 4)
        self.head = nn.Linear(4, rows, bias=False)
        with torch.no_grad():
            self.embed.weight[:len(old_e)].copy_(old_e)
            self.head.weight[:len(old_h)].copy_(old_h)
        if self.tied:
            self.head.weight = self.embed.weight


@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize("rows", [8, 12])
def test_expansion_preserves_rows_and_resume(tied, rows):
    model, tokenizer = Model(tied, rows), Tokenizer()
    before = model.embed.weight[:8].detach().clone()
    before_head = model.head.weight[:8].detach().clone()
    report = expand_action_vocabulary(tokenizer, model, count=3)
    assert report["token_ids"] == [8, 10]
    torch.testing.assert_close(model.embed.weight[:8], before, rtol=0, atol=0)
    torch.testing.assert_close(model.embed.weight[8:11], before.mean(0).expand(3, -1))
    torch.testing.assert_close(model.head.weight[8:11], before_head.mean(0).expand(3, -1))
    assert (model.embed.weight.data_ptr() == model.head.weight.data_ptr()) == tied
    data = io.BytesIO()
    torch.save(model.state_dict(), data)
    data.seek(0)
    restored = Model(tied, len(model.embed.weight))
    restored.load_state_dict(torch.load(data, weights_only=True))
    assert expand_action_vocabulary(tokenizer, restored, count=3)["added"] == 0
    torch.testing.assert_close(model.head.weight, restored.head.weight)


def test_partial_vocabulary_rejected():
    tokenizer = Tokenizer()
    tokenizer.vocab["<robot_action_0>"] = 8
    with pytest.raises(ValueError, match="partial"):
        expand_action_vocabulary(tokenizer, Model(True), count=3)
