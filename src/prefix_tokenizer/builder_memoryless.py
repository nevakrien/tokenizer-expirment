from __future__ import annotations

import heapq
import math
from collections import Counter

from .tree import Trie, TrieNode


def build_memoryless_tree(
    data: bytes,
    phrase_leaf_count: int,
    alpha: float = 1.0,
    max_depth: int = 64,
    show_progress: bool = False,
) -> Trie:
    if phrase_leaf_count < 256 or (phrase_leaf_count - 256) % 255 != 0:
        raise ValueError("phrase_leaf_count must be 256 + 255*k")
    counts = Counter(data)
    total = len(data) + 256 * alpha
    log_probs = [math.log((counts[byte] + alpha) / total) for byte in range(256)]
    trie = Trie.initial_bytes()
    heap: list[tuple[float, int, bytes, int, TrieNode]] = []
    for node in trie.leaves():
        score = log_probs[node.phrase[-1]]
        heapq.heappush(heap, (-score, len(node.phrase), node.phrase, node.creation_index, node))
    leaf_count = 256
    progress = None
    if show_progress:
        from tqdm.auto import tqdm

        progress = tqdm(total=(phrase_leaf_count - 256) // 255, desc="Building memoryless tree", unit="expansion")
    while leaf_count + 255 <= phrase_leaf_count:
        if not heap:
            break
        neg_score, _, _, _, node = heapq.heappop(heap)
        if not node.is_leaf or len(node.phrase) >= max_depth:
            continue
        score = -neg_score
        trie.expand_leaf(node)
        assert node.children is not None
        for byte, child in enumerate(node.children):
            child_score = score + log_probs[byte]
            heapq.heappush(heap, (-child_score, len(child.phrase), child.phrase, child.creation_index, child))
        leaf_count += 255
        if progress is not None:
            progress.update(1)
    if progress is not None:
        progress.close()
    return trie
