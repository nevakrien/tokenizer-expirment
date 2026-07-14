import json

from experiment.analyze_tokenizers import main
from prefix_tokenizer import PrefixTreeTokenizer, build_corpus_count_tree, compute_vocab_layout


def test_analyze_tokenizers_reports_configured_groups(tmp_path, monkeypatch) -> None:
    english = tmp_path / "english.txt"
    hebrew = tmp_path / "hebrew.txt"
    english.write_bytes(b"hello world hello")
    hebrew.write_bytes("שלום עולם שלום".encode())
    config = tmp_path / "data.json"
    config.write_text(
        json.dumps(
            {
                "files": [
                    {"path": str(english), "language": "english"},
                    {"path": str(hebrew), "language": "hebrew"},
                ]
            }
        ),
        encoding="utf-8",
    )

    docs = [english.read_bytes(), hebrew.read_bytes(), bytes(range(256))]
    layout = compute_vocab_layout(1025)
    trie = build_corpus_count_tree(docs, layout.phrase_leaf_count, expansion_batch_size=2, max_depth=8)
    tokenizer_path = tmp_path / "tokenizer"
    PrefixTreeTokenizer.from_trie(trie, vocab_size=1025).save_pretrained(tokenizer_path)

    output = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_tokenizers",
            "--tokenizers",
            str(tokenizer_path),
            "--dataset-config",
            str(config),
            "--output",
            str(output),
        ],
    )
    main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["groups"] == {"english": {"document_count": 1}, "hebrew": {"document_count": 1}}
    tokenizer_report = report["tokenizers"][0]
    assert set(tokenizer_report["groups"]) == {"english", "hebrew"}
    assert tokenizer_report["groups"]["english"]["bytes_per_token"] > 0
    assert tokenizer_report["groups"]["hebrew"]["unicode_chars_per_token"] > 0
