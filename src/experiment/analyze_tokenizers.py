from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from .common import load_documents
from .reference_bpe import ReferenceBPETokenizer, load_any_tokenizer


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
        tokenizer = load_any_tokenizer(tokenizer_path)
        is_bpe = isinstance(tokenizer, ReferenceBPETokenizer)
        if is_bpe:
            active_count = len(tokenizer._token_by_id)  # noqa: SLF001
        else:
            active_count = len(tokenizer.phrase_token_ids)
        token_lengths = []
        phrase_lengths = Counter()
        token_counts = Counter()
        pair_counts = Counter()
        total_tokens = 0
        observed_ids = set()
        total_bytes = sum(len(document) for document in documents)
        for document in documents:
            ids = tokenizer.encode_bytes(document)
            token_lengths.append(len(ids))
            total_tokens += len(ids)
            observed_ids.update(ids)
            token_counts.update(ids)
            pair_counts.update(zip(ids, ids[1:], strict=False))
            for token_id in ids:
                phrase = getattr(tokenizer, "_phrase_by_id", {}).get(token_id)
                if phrase is not None:
                    phrase_lengths[len(phrase)] += 1
                elif is_bpe and token_id in tokenizer._token_by_id:  # noqa: SLF001
                    phrase_lengths[len(tokenizer.decode_bytes([token_id]))] += 1
        active_utilization = len(observed_ids) / active_count if active_count else 0.0
        report["tokenizers"].append(
            {
                "path": tokenizer_path,
                "model_vocab_size": tokenizer.vocab_size,
                "type": "byte_bpe" if is_bpe else "byte_prefix_tree",
                "active_phrases": active_count,
                "reserved_ids": len(tokenizer.reserved_ids or []),
                "observed_ids": len(observed_ids),
                "active_id_utilization": active_utilization,
                "unique_observed_pairs": len(pair_counts),
                "pair_utilization": _pair_utilization(len(pair_counts), tokenizer.vocab_size),
                "pair_entropy_bits": _entropy_bits(pair_counts),
                "unigram_entropy_bits": _entropy_bits(token_counts),
                "top_token_frequency": _top_token_frequency(token_counts, total_tokens),
                "tokens_covering_50_percent": _tokens_covering(token_counts, 0.50),
                "tokens_covering_90_percent": _tokens_covering(token_counts, 0.90),
                "tokens_covering_99_percent": _tokens_covering(token_counts, 0.99),
                "zero_frequency_active_ids": max(active_count - len(observed_ids), 0),
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


def _entropy_bits(counts: Counter) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def _pair_utilization(pair_count: int, vocab_size: int) -> float:
    return pair_count / (vocab_size * vocab_size) if vocab_size else 0.0


def _top_token_frequency(counts: Counter, total_tokens: int) -> float:
    return max(counts.values(), default=0) / total_tokens if total_tokens else 0.0


def _tokens_covering(counts: Counter, fraction: float) -> int:
    total = sum(counts.values())
    if not total:
        return 0
    threshold = total * fraction
    cumulative = 0
    for index, count in enumerate(sorted(counts.values(), reverse=True), start=1):
        cumulative += count
        if cumulative >= threshold:
            return index
    return len(counts)


if __name__ == "__main__":
    main()
