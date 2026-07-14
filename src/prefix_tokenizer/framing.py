from __future__ import annotations

from .tokenizer import PrefixTreeTokenizer


def encode_documents(tokenizer: PrefixTreeTokenizer, documents: list[bytes]) -> list[list[int]]:
    return [tokenizer.encode_bytes(document, add_eos=True) for document in documents]


def pack_token_blocks(encoded_documents: list[list[int]], context_length: int) -> list[list[int]]:
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    stream: list[int] = []
    for document in encoded_documents:
        stream.extend(document)
    return [stream[index : index + context_length] for index in range(0, len(stream), context_length) if len(stream[index : index + context_length]) == context_length]
