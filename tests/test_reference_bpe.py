import pytest

from experiment.reference_bpe import ReferenceBPETokenizer


def test_reference_bpe_roundtrip_and_reserved_ids(tmp_path) -> None:
    pytest.importorskip("tokenizers")

    docs = [b"hello hello", bytes(range(256)), "multilingual שלום".encode()]
    tokenizer = ReferenceBPETokenizer.train(docs, vocab_size=300)
    tokenizer.save_pretrained(tmp_path)
    loaded = ReferenceBPETokenizer.from_pretrained(tmp_path)

    for data in docs + [b"\x00\xffhello"]:
        assert loaded.decode_bytes(loaded.encode_bytes(data)) == data

    if loaded.reserved_ids:
        with pytest.raises(ValueError, match="reserved token id"):
            loaded.decode_bytes([min(loaded.reserved_ids)])
