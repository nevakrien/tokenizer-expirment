from __future__ import annotations

from .tokenizer import PrefixTreeTokenizer


class PrefixTreeHFAdapter:
    """Small compatibility wrapper for code that expects tokenizer-like callables."""

    def __init__(self, tokenizer: PrefixTreeTokenizer) -> None:
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.unk_token_id = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        input_ids = self.encode(text, add_special_tokens=add_special_tokens)
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
