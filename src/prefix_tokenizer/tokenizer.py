from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .tree import Trie, assign_token_ids


@dataclass
class PrefixTreeTokenizer:
    trie: Trie
    vocab_size: int
    eos_token_id: int
    bos_token_id: int | None = None
    pad_token_id: int | None = None
    tail_token_start: int | None = None
    reserved_ids: set[int] | None = None

    def __post_init__(self) -> None:
        self._phrase_by_id = {node.token_id: node.phrase for node in self.trie.leaves() if node.token_id is not None}
        self._reserved_ids = self.reserved_ids or set()

    @classmethod
    def from_trie(
        cls,
        trie: Trie,
        vocab_size: int,
        special_token_count: int = 1,
        tail_token_count: int = 256,
    ) -> "PrefixTreeTokenizer":
        leaves = assign_token_ids(trie)
        next_id = len(leaves)
        tail_token_start = next_id if tail_token_count else None
        next_id += tail_token_count
        eos_token_id = next_id
        next_id += special_token_count
        reserved_ids = set(range(next_id, vocab_size))
        tokenizer = cls(
            trie=trie,
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            tail_token_start=tail_token_start,
            reserved_ids=reserved_ids,
        )
        tokenizer.trie.validate(model_vocab_size=vocab_size, reserved_ids=reserved_ids)
        return tokenizer

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.encode_bytes(text.encode("utf-8"), add_eos=add_special_tokens)

    def encode_bytes(self, data: bytes, add_eos: bool = False) -> list[int]:
        output: list[int] = []
        position = 0
        while position < len(data):
            node = self.trie.root
            start = position
            while node.children is not None:
                if position >= len(data):
                    self._append_tail_bytes(output, data[start:])
                    if add_eos:
                        output.append(self.eos_token_id)
                    return output
                node = node.children[data[position]]
                position += 1
            if node.token_id is None:
                raise RuntimeError("reached leaf without token_id")
            output.append(node.token_id)
        if add_eos:
            output.append(self.eos_token_id)
        return output

    def _append_tail_bytes(self, output: list[int], data: bytes) -> None:
        if not data:
            return
        if self.tail_token_start is None:
            raise ValueError("input ended inside an internal node and tail tokens are disabled")
        output.extend(self.tail_token_start + byte for byte in data)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return self.decode_bytes(token_ids, skip_special_tokens=skip_special_tokens).decode("utf-8", errors="strict")

    def decode_bytes(self, token_ids: list[int], skip_special_tokens: bool = False) -> bytes:
        output = bytearray()
        for token_id in token_ids:
            if token_id == self.eos_token_id or token_id in {self.bos_token_id, self.pad_token_id}:
                if not skip_special_tokens:
                    continue
                continue
            if self.tail_token_start is not None and self.tail_token_start <= token_id < self.tail_token_start + 256:
                output.append(token_id - self.tail_token_start)
                continue
            if token_id in self._reserved_ids:
                raise ValueError(f"reserved token id cannot be decoded: {token_id}")
            phrase = self._phrase_by_id.get(token_id)
            if phrase is None:
                raise ValueError(f"unknown token id: {token_id}")
            output.extend(phrase)
        return bytes(output)

    @property
    def phrase_token_ids(self) -> list[int]:
        return sorted(token_id for token_id in self._phrase_by_id if token_id is not None)

    def save_pretrained(self, path: str | Path) -> None:
        from .serialization import save_tokenizer

        save_tokenizer(self, Path(path))

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "PrefixTreeTokenizer":
        from .serialization import load_tokenizer

        return load_tokenizer(Path(path))
