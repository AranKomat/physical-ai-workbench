"""Strict FAST label checks; no normalization or donor slot conversion is implicit."""
import numpy as np


def checked_fast_labels(processor, actions: np.ndarray) -> dict:
    from scipy.fft import dct, idct

    actions = np.asarray(actions)
    if actions.ndim != 2 or not actions.size or not np.isfinite(actions).all():
        raise ValueError("finite nonempty [time, dimension] actions required")
    expected = np.rint(dct(actions, axis=0, norm="ortho") * processor.scale).astype(np.int64)
    if expected.min() < processor.min_token:
        raise ValueError("FAST would clip DCT coefficients")
    codes = processor(actions)[0]
    if not codes or any(not isinstance(i, (int, np.integer)) or i < 0 or i >= processor.vocab_size for i in codes):
        raise ValueError("invalid FAST codes")
    # Verify BPE losslessness before upstream decode can silently return zeros.
    decoded_text = processor.bpe_tokenizer.decode(codes)
    recovered = np.asarray([ord(c) + processor.min_token for c in decoded_text])
    if not np.array_equal(recovered, expected.ravel()):
        raise ValueError("FAST BPE round trip changed quantized coefficients")
    restored = np.asarray(processor.decode([codes], time_horizon=actions.shape[0],
                                          action_dim=actions.shape[1]))[0]
    if restored.shape != actions.shape or not np.isfinite(restored).all():
        raise ValueError("invalid decoded FAST actions")
    if not np.allclose(restored, idct(expected / processor.scale, axis=0, norm="ortho"), atol=1e-6):
        raise ValueError("decoded FAST actions differ from checked coefficients")
    error = restored - actions
    return {"codes": [int(i) for i in codes],
            "text": "".join(f"<robot_action_{i}>" for i in codes),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "per_dimension_rmse": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
            "max_absolute_error": float(np.max(np.abs(error))),
            "shape": list(actions.shape)}
