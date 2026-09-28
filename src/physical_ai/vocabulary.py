"""Append robot-action vocabulary before constructing an optimizer."""
import inspect
import torch


def expand_action_vocabulary(tokenizer, model, count=2048):
    if count < 1:
        raise ValueError("positive action vocabulary size required")
    names = [f"<robot_action_{i}>" for i in range(count)]
    vocab = tokenizer.get_vocab()
    present = [name in vocab for name in names]
    if any(present) and not all(present):
        raise ValueError("partial action vocabulary")
    old_vocab = len(tokenizer)
    embed = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    tied = embed.data_ptr() == head.data_ptr()
    if all(present):
        ids = tokenizer.convert_tokens_to_ids(names)
        if ids != list(range(ids[0], ids[0] + count)) or max(ids) >= min(len(embed), len(head)):
            raise ValueError("existing action vocabulary and model disagree")
        return {"added": 0, "token_ids": [ids[0], ids[-1]], "tied": tied}
    if len(embed) < old_vocab or len(head) < old_vocab:
        raise ValueError("model does not cover original tokenizer")
    # Exclude unused model padding rows from the semantic initialization mean.
    input_mean = embed[:old_vocab].detach().mean(dim=0, keepdim=True).clone()
    output_mean = head[:old_vocab].detach().mean(dim=0, keepdim=True).clone()
    tokenizer.add_special_tokens({"additional_special_tokens": names})
    ids = tokenizer.convert_tokens_to_ids(names)
    if ids != list(range(old_vocab, old_vocab + count)):
        raise ValueError("action tokens were not appended contiguously")
    options = {"mean_resizing": False} if "mean_resizing" in inspect.signature(model.resize_token_embeddings).parameters else {}
    model.resize_token_embeddings(max(len(embed), len(tokenizer)), **options)
    embed = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    if (embed.data_ptr() == head.data_ptr()) != tied:
        raise ValueError("vocabulary resize changed weight tying")
    with torch.no_grad():
        embed[old_vocab:old_vocab + count].copy_(input_mean)
        if not tied:
            head[old_vocab:old_vocab + count].copy_(output_mean)
    return {"added": count, "token_ids": [ids[0], ids[-1]], "tied": tied,
            "embedding_rows": len(embed), "initialization": "mean of preexisting tokenizer rows"}
