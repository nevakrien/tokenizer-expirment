from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from prefix_tokenizer import PrefixTreeTokenizer

from .common import load_documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizers", nargs="+", required=True)
    parser.add_argument("--dataset-config")
    parser.add_argument("--input-files", nargs="*")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    documents = load_documents(args.input_files, args.dataset_config)
    report = {"document_count": len(documents), "tokenizers": []}
    for tokenizer_path in args.tokenizers:
        tokenizer = PrefixTreeTokenizer.from_pretrained(tokenizer_path)
        token_lengths = []
        phrase_lengths = Counter()
        total_tokens = 0
        total_bytes = sum(len(document) for document in documents)
        for document in documents:
            ids = tokenizer.encode_bytes(document)
            token_lengths.append(len(ids))
            total_tokens += len(ids)
            for token_id in ids:
                phrase = tokenizer._phrase_by_id.get(token_id)  # noqa: SLF001
                if phrase is not None:
                    phrase_lengths[len(phrase)] += 1
        report["tokenizers"].append(
            {
                "path": tokenizer_path,
                "model_vocab_size": tokenizer.vocab_size,
                "active_phrases": len(tokenizer.phrase_token_ids),
                "reserved_ids": len(tokenizer.reserved_ids or []),
                "bytes_per_token": total_bytes / total_tokens if total_tokens else 0.0,
                "tokens_per_byte": total_tokens / total_bytes if total_bytes else 0.0,
                "tokens_per_document_mean": total_tokens / len(documents) if documents else 0.0,
                "tokens_per_document_median": sorted(token_lengths)[len(token_lengths) // 2] if token_lengths else 0,
                "phrase_length_histogram": dict(sorted(phrase_lengths.items())),
                "maximum_phrase_length": max(phrase_lengths, default=0),
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
