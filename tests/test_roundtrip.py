import random

from prefix_tokenizer import PrefixTreeTokenizer, build_corpus_count_tree, compute_vocab_layout


def make_tokenizer(vocab_size: int = 1025) -> PrefixTreeTokenizer:
    docs = [b"abracadabra abracadabra", bytes(range(256)), b"aaaaabbbbcccdde"]
    layout = compute_vocab_layout(vocab_size)
    trie = build_corpus_count_tree(docs, layout.phrase_leaf_count, expansion_batch_size=2, max_depth=8)
    return PrefixTreeTokenizer.from_trie(trie, vocab_size=vocab_size)


def test_exhaustive_lengths_zero_one_two_roundtrip() -> None:
    tokenizer = make_tokenizer()
    assert tokenizer.decode_bytes(tokenizer.encode_bytes(b"")) == b""
    for value in range(256):
        data = bytes([value])
        assert tokenizer.decode_bytes(tokenizer.encode_bytes(data)) == data
    for value in range(256 * 256):
        data = bytes([value >> 8, value & 0xFF])
        assert tokenizer.decode_bytes(tokenizer.encode_bytes(data)) == data


def test_random_roundtrip_cases() -> None:
    tokenizer = make_tokenizer()
    rng = random.Random(123)
    cases = [b"\x00" * 100, bytes(range(256)), "hello pi lambda".encode()]
    cases.extend(bytes(rng.randrange(256) for _ in range(rng.randrange(128))) for _ in range(1000))
    for data in cases:
        assert tokenizer.decode_bytes(tokenizer.encode_bytes(data)) == data


def test_text_uses_strict_utf8() -> None:
    tokenizer = make_tokenizer()
    text = "hello pi lambda"
    assert tokenizer.decode(tokenizer.encode(text)) == text
