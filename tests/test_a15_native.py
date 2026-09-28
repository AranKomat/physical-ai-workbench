import pytest
import torch
from torch import nn

from physical_ai.a15_native import A15NativeCacheAdapter


class Native(nn.Module):
    def denoise_step_full(self, state, mask, cache, positions, x, t, fast_mask):
        assert fast_mask is None
        return x * 2, None


def inputs():
    return dict(state=torch.zeros(1, 32), prefix_ids=torch.tensor([[1, 2]]),
                prefix_mask=torch.ones(1, 2, dtype=torch.bool), cache=object(),
                max_prefix_position_ids=torch.zeros(3, 1, 1, dtype=torch.long),
                noisy_actions=torch.ones(1, 50, 32), native_time=torch.tensor([.5]))


def test_a15_native_cache_boundary():
    result = A15NativeCacheAdapter(Native())(**inputs())
    torch.testing.assert_close(result, torch.full((1, 50, 32), 2.))


def test_a15_rejects_answer_tokens_and_missing_cache():
    args = inputs()
    args["prefix_ids"][0, 0] = 248077
    with pytest.raises(ValueError, match="answer"):
        A15NativeCacheAdapter(Native())(**args)
    args = inputs()
    args["cache"] = None
    with pytest.raises(ValueError, match="cache"):
        A15NativeCacheAdapter(Native())(**args)


def test_a15_rejects_research_shape():
    args = inputs()
    args["noisy_actions"] = torch.zeros(1, 30, 80)
    with pytest.raises(ValueError, match="shape"):
        A15NativeCacheAdapter(Native())(**args)
