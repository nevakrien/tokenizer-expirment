from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .tokenizer import PrefixTreeTokenizer
from .tree import Trie


FORMAT_VERSION = 1


def corpus_sha256(documents: list[bytes]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(len(document).to_bytes(8, "little"))
        digest.update(document)
    return digest.hexdigest()


def save_tokenizer(tokenizer: PrefixTreeTokenizer, path: Path, *, metadata: dict | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    phrases = sorted(tokenizer.trie.leaves(), key=lambda node: node.token_id if node.token_id is not None else -1)
    phrase_hex = [node.phrase.hex() for node in phrases]
    (path / "tokenizer.json").write_text(
        json.dumps(
            {
                "format_version": FORMAT_VERSION,
                "vocab_size": tokenizer.vocab_size,
                "eos_token_id": tokenizer.eos_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "tail_token_start": tokenizer.tail_token_start,
                "reserved_ids": sorted(tokenizer.reserved_ids or []),
                "phrases_hex": phrase_hex,
                "corpus_counts": [node.corpus_count for node in phrases],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (path / "phrases.bin").write_bytes(b"".join(len(bytes.fromhex(item)).to_bytes(2, "little") + bytes.fromhex(item) for item in phrase_hex))
    (path / "tree.bin").write_text(json.dumps({"phrases_hex": phrase_hex}), encoding="utf-8")
    base_metadata = {
        "type": "byte_prefix_tree",
        "format_version": FORMAT_VERSION,
        "model_vocab_size": tokenizer.vocab_size,
        "phrase_leaf_count": len(phrase_hex),
        "tail_token_count": 256 if tokenizer.tail_token_start is not None else 0,
        "special_token_count": 1 + int(tokenizer.bos_token_id is not None) + int(tokenizer.pad_token_id is not None),
        "reserved_token_count": len(tokenizer.reserved_ids or []),
        "maximum_phrase_bytes": tokenizer.trie.max_depth(),
        "average_phrase_bytes_training": _average_phrase_bytes(tokenizer),
    }
    if metadata:
        base_metadata.update(metadata)
    (path / "metadata.json").write_text(json.dumps(base_metadata, indent=2, sort_keys=True), encoding="utf-8")


def load_tokenizer(path: Path) -> PrefixTreeTokenizer:
    payload = json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))
    phrases = [bytes.fromhex(item) for item in payload["phrases_hex"]]
    trie = Trie.from_phrases(phrases)
    by_phrase = {node.phrase: node for node in trie.leaves()}
    for token_id, phrase in enumerate(phrases):
        node = by_phrase[phrase]
        node.token_id = token_id
        node.corpus_count = payload.get("corpus_counts", [0] * len(phrases))[token_id]
    return PrefixTreeTokenizer(
        trie=trie,
        vocab_size=payload["vocab_size"],
        eos_token_id=payload["eos_token_id"],
        bos_token_id=payload.get("bos_token_id"),
        pad_token_id=payload.get("pad_token_id"),
        tail_token_start=payload.get("tail_token_start"),
        reserved_ids=set(payload.get("reserved_ids", [])),
    )


def _average_phrase_bytes(tokenizer: PrefixTreeTokenizer) -> float:
    total_count = 0
    total_bytes = 0
    for node in tokenizer.trie.leaves():
        total_count += node.corpus_count
        total_bytes += node.corpus_count * len(node.phrase)
    return total_bytes / total_count if total_count else 0.0
