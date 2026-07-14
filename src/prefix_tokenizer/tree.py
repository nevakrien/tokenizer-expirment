from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class VocabLayout:
    model_vocab_size: int
    phrase_leaf_count: int
    tail_token_count: int
    special_token_count: int
    reserved_token_count: int


def compute_vocab_layout(
    model_vocab_size: int,
    special_token_count: int = 1,
    tail_token_count: int = 256,
) -> VocabLayout:
    available = model_vocab_size - special_token_count - tail_token_count
    if available < 256:
        raise ValueError("vocab size is too small for 256 byte leaves plus specials/tails")
    expansions = (available - 256) // 255
    phrase_leaf_count = 256 + 255 * expansions
    reserved_token_count = available - phrase_leaf_count
    return VocabLayout(
        model_vocab_size=model_vocab_size,
        phrase_leaf_count=phrase_leaf_count,
        tail_token_count=tail_token_count,
        special_token_count=special_token_count,
        reserved_token_count=reserved_token_count,
    )


@dataclass(eq=False)
class TrieNode:
    phrase: bytes
    creation_index: int
    children: list["TrieNode"] | None = None
    token_id: int | None = None
    corpus_count: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.children is None


@dataclass
class Trie:
    root: TrieNode
    next_creation_index: int = 0

    @classmethod
    def initial_bytes(cls) -> "Trie":
        root = TrieNode(phrase=b"", creation_index=0, children=[])
        trie = cls(root=root, next_creation_index=1)
        root.children = [trie._new_node(bytes([byte])) for byte in range(256)]
        return trie

    @classmethod
    def from_phrases(cls, phrases: Iterable[bytes]) -> "Trie":
        trie = cls.initial_empty_root()
        for phrase in sorted(phrases, key=lambda item: (len(item), item)):
            trie.insert_leaf(phrase)
        return trie

    @classmethod
    def initial_empty_root(cls) -> "Trie":
        return cls(root=TrieNode(phrase=b"", creation_index=0, children=[None] * 256), next_creation_index=1)  # type: ignore[list-item]

    def _new_node(self, phrase: bytes) -> TrieNode:
        node = TrieNode(phrase=phrase, creation_index=self.next_creation_index)
        self.next_creation_index += 1
        return node

    def expand_leaf(self, node: TrieNode) -> None:
        if not node.is_leaf:
            raise ValueError("cannot expand an internal node")
        node.token_id = None
        node.children = [self._new_node(node.phrase + bytes([byte])) for byte in range(256)]

    def insert_leaf(self, phrase: bytes) -> None:
        if not phrase:
            raise ValueError("empty phrase cannot be a leaf")
        node = self.root
        for offset, byte in enumerate(phrase):
            if node.children is None:
                node.children = [None] * 256  # type: ignore[list-item]
                node.token_id = None
            child = node.children[byte]
            if child is None:
                child = self._new_node(phrase[: offset + 1])
                node.children[byte] = child
            node = child
        node.children = None

    def leaves(self) -> list[TrieNode]:
        output: list[TrieNode] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.children is None:
                output.append(node)
            else:
                stack.extend(reversed(node.children))
        return output

    def internal_nodes(self) -> list[TrieNode]:
        output: list[TrieNode] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.children is not None:
                output.append(node)
                stack.extend(reversed(node.children))
        return output

    def max_depth(self) -> int:
        return max((len(node.phrase) for node in self.leaves()), default=0)

    def validate(self, model_vocab_size: int | None = None, reserved_ids: set[int] | None = None) -> None:
        seen_ids: set[int] = set()
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.children is None:
                if node.token_id is None:
                    raise ValueError(f"leaf {node.phrase!r} has no token_id")
                if node.token_id in seen_ids:
                    raise ValueError(f"duplicate token_id {node.token_id}")
                if reserved_ids and node.token_id in reserved_ids:
                    raise ValueError(f"reserved token_id {node.token_id} assigned to phrase")
                if model_vocab_size is not None and not 0 <= node.token_id < model_vocab_size:
                    raise ValueError(f"token_id {node.token_id} outside model vocab")
                seen_ids.add(node.token_id)
            else:
                if len(node.children) != 256:
                    raise ValueError("internal node does not have exactly 256 children")
                if any(child is None for child in node.children):
                    raise ValueError("internal node has missing child")
                stack.extend(node.children)


def assign_token_ids(trie: Trie, start_id: int = 0) -> list[TrieNode]:
    leaves = trie.leaves()
    leaves.sort(key=lambda node: (-node.corpus_count, -len(node.phrase), node.phrase))
    for token_id, node in enumerate(leaves, start=start_id):
        node.token_id = token_id
    return leaves
