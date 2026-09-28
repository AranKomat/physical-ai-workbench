from types import SimpleNamespace

import numpy as np
import pytest

from physical_ai.action_labels import checked_fast_labels


class ZeroCodec:
    scale = 10
    min_token = -354
    vocab_size = 2048
    bpe_tokenizer = SimpleNamespace(decode=lambda codes: chr(354) * 6)

    def __call__(self, actions):
        return [[0, 1]]

    def decode(self, codes, **kwargs):
        return np.zeros((1, kwargs["time_horizon"], kwargs["action_dim"]))


def test_checked_fast_labels():
    result = checked_fast_labels(ZeroCodec(), np.zeros((3, 2)))
    assert result["text"] == "<robot_action_0><robot_action_1>"
    assert result["rmse"] == 0


@pytest.mark.parametrize("actions", [np.zeros(3), np.zeros((0, 2)), np.full((3, 2), np.nan)])
def test_invalid_actions(actions):
    with pytest.raises(ValueError, match="finite nonempty"):
        checked_fast_labels(ZeroCodec(), actions)


def test_reject_clipping_and_corrupted_codes():
    with pytest.raises(ValueError, match="clip"):
        checked_fast_labels(ZeroCodec(), np.full((3, 2), -1000.0))
    codec = ZeroCodec()
    codec.bpe_tokenizer = SimpleNamespace(decode=lambda codes: "bad")
    with pytest.raises(ValueError, match="round trip"):
        checked_fast_labels(codec, np.zeros((3, 2)))


def test_reject_nonfinite_decode():
    codec = ZeroCodec()
    codec.decode = lambda *args, **kwargs: np.full((1, 3, 2), np.nan)
    with pytest.raises(ValueError, match="invalid decoded"):
        checked_fast_labels(codec, np.zeros((3, 2)))


def test_reject_silent_zero_decode():
    from scipy.fft import idct

    codec = ZeroCodec()
    codec.bpe_tokenizer = SimpleNamespace(decode=lambda codes: chr(355) * 6)
    actions = idct(np.ones((3, 2)) / 10, axis=0, norm="ortho")
    with pytest.raises(ValueError, match="differ from checked"):
        checked_fast_labels(codec, actions)
