from prefix_tokenizer import PrefixTreeTokenizer, build_memoryless_tree, compute_vocab_layout


def test_memoryless_tree_invariants() -> None:
    layout = compute_vocab_layout(1025)
    trie = build_memoryless_tree(b"hello world" * 10, layout.phrase_leaf_count, max_depth=8)
    tokenizer = PrefixTreeTokenizer.from_trie(trie, vocab_size=1025)
    tokenizer.trie.validate(model_vocab_size=tokenizer.vocab_size, reserved_ids=tokenizer.reserved_ids)
    assert len(tokenizer.phrase_token_ids) == layout.phrase_leaf_count
    assert len(tokenizer.reserved_ids or []) == layout.reserved_token_count


def test_reserved_ids_rejected() -> None:
    layout = compute_vocab_layout(1026)
    trie = build_memoryless_tree(b"abc", layout.phrase_leaf_count, max_depth=4)
    tokenizer = PrefixTreeTokenizer.from_trie(trie, vocab_size=1026)
    reserved = sorted(tokenizer.reserved_ids or [])
    assert reserved
    try:
        tokenizer.decode_bytes([reserved[0]])
    except ValueError as exc:
        assert "reserved" in str(exc)
    else:
        raise AssertionError("reserved id decoded successfully")
