from prefix_tokenizer import PrefixTreeTokenizer, build_corpus_count_tree, compute_vocab_layout


def test_save_and_load_roundtrip(tmp_path) -> None:
    docs = [b"the quick brown fox", b"the quick blue fox"]
    layout = compute_vocab_layout(1025)
    trie = build_corpus_count_tree(docs, layout.phrase_leaf_count, expansion_batch_size=2, max_depth=8)
    tokenizer = PrefixTreeTokenizer.from_trie(trie, vocab_size=1025)
    tokenizer.save_pretrained(tmp_path)
    loaded = PrefixTreeTokenizer.from_pretrained(tmp_path)
    for data in [b"the quick", b"fox", bytes(range(64))]:
        assert loaded.encode_bytes(data) == tokenizer.encode_bytes(data)
        assert loaded.decode_bytes(loaded.encode_bytes(data)) == data


def test_translation_special_tokens_survive_roundtrip(tmp_path) -> None:
    docs = [b"hello world"]
    layout = compute_vocab_layout(1025, special_token_count=3)
    trie = build_corpus_count_tree(docs, layout.phrase_leaf_count, expansion_batch_size=2, max_depth=8)
    tokenizer = PrefixTreeTokenizer.from_trie(trie, vocab_size=1025, special_token_count=3)
    tokenizer.save_pretrained(tmp_path)
    loaded = PrefixTreeTokenizer.from_pretrained(tmp_path)

    assert (loaded.eos_token_id, loaded.bos_token_id, loaded.pad_token_id) == (
        tokenizer.eos_token_id,
        tokenizer.bos_token_id,
        tokenizer.pad_token_id,
    )
