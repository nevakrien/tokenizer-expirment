from __future__ import annotations

from collections import Counter

from .tree import Trie, TrieNode


def _scan_documents(trie: Trie, documents: list[bytes]) -> Counter[TrieNode]:
    counts: Counter[TrieNode] = Counter()
    for data in documents:
        position = 0
        while position < len(data):
            node = trie.root
            start = position
            while node.children is not None and position < len(data):
                node = node.children[data[position]]
                position += 1
            if node.children is not None:
                break
            counts[node] += 1
            if position == start:
                raise RuntimeError("encoder made no progress")
    return counts


def build_corpus_count_tree(
    documents: list[bytes],
    phrase_leaf_count: int,
    expansion_batch_size: int = 64,
    max_depth: int = 64,
) -> Trie:
    if phrase_leaf_count < 256 or (phrase_leaf_count - 256) % 255 != 0:
        raise ValueError("phrase_leaf_count must be 256 + 255*k")
    trie = Trie.initial_bytes()
    leaf_count = 256
    batch_size = max(1, expansion_batch_size)
    while leaf_count + 255 <= phrase_leaf_count:
        counts = _scan_documents(trie, documents)
        for node in trie.leaves():
            node.corpus_count = counts.get(node, 0)
        candidates = [node for node in trie.leaves() if len(node.phrase) < max_depth]
        candidates.sort(key=lambda node: (-node.corpus_count, len(node.phrase), node.phrase, node.creation_index))
        remaining_expansions = (phrase_leaf_count - leaf_count) // 255
        expanded = 0
        for node in candidates[: min(batch_size, remaining_expansions)]:
            if node.is_leaf:
                trie.expand_leaf(node)
                expanded += 1
        if expanded == 0:
            break
        leaf_count += 255 * expanded
    counts = _scan_documents(trie, documents)
    for node in trie.leaves():
        node.corpus_count = counts.get(node, 0)
    return trie
