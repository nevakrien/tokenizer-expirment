from itertools import product

from prefix_tokenizer import PrefixTreeTokenizer, build_memoryless_tree, compute_vocab_layout


def test_phrase_sequences_decode_uniquely_up_to_length_two() -> None:
    layout = compute_vocab_layout(1025)
    trie = build_memoryless_tree(b"abcabcabc", layout.phrase_leaf_count, max_depth=4)
    tokenizer = PrefixTreeTokenizer.from_trie(trie, vocab_size=1025)
    ids = tokenizer.phrase_token_ids[:32]
    decoded: dict[bytes, tuple[int, ...]] = {}
    for length in [1, 2]:
        for sequence in product(ids, repeat=length):
            data = tokenizer.decode_bytes(list(sequence))
            previous = decoded.get(data)
            assert previous is None or previous == sequence
            decoded[data] = sequence
