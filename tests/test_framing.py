from prefix_tokenizer import PrefixTreeTokenizer, build_corpus_count_tree, compute_vocab_layout
from prefix_tokenizer.framing import encode_documents, pack_token_blocks


def test_tail_tokens_allow_internal_eof() -> None:
    layout = compute_vocab_layout(1025)
    trie = build_corpus_count_tree([b"aa" * 100], layout.phrase_leaf_count, expansion_batch_size=2, max_depth=8)
    tokenizer = PrefixTreeTokenizer.from_trie(trie, vocab_size=1025)
    encoded = tokenizer.encode_bytes(b"a", add_eos=True)
    assert encoded[-1] == tokenizer.eos_token_id
    assert tokenizer.decode_bytes(encoded) == b"a"


def test_document_packing() -> None:
    layout = compute_vocab_layout(1025)
    trie = build_corpus_count_tree([b"abcdef"], layout.phrase_leaf_count, expansion_batch_size=1, max_depth=4)
    tokenizer = PrefixTreeTokenizer.from_trie(trie, vocab_size=1025)
    encoded = encode_documents(tokenizer, [b"abc", b"def"])
    assert all(document[-1] == tokenizer.eos_token_id for document in encoded)
    blocks = pack_token_blocks(encoded, context_length=2)
    assert all(len(block) == 2 for block in blocks)
