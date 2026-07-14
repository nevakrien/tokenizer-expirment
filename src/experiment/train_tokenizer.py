from __future__ import annotations

import argparse
from pathlib import Path

from prefix_tokenizer import PrefixTreeTokenizer, build_corpus_count_tree, build_memoryless_tree, compute_vocab_layout
from prefix_tokenizer.serialization import corpus_sha256, save_tokenizer

from .common import load_documents
from .reference_bpe import ReferenceBPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["prefix_tree", "memoryless", "bpe"], default="prefix_tree")
    parser.add_argument("--algorithm", choices=["corpus_count_batched", "memoryless"], default="corpus_count_batched")
    parser.add_argument("--dataset-config")
    parser.add_argument("--input-files", nargs="*")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expansion-batch-size", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=64)
    parser.add_argument("--tail-byte-tokens", action="store_true", default=True)
    args = parser.parse_args()

    documents = load_documents(args.input_files, args.dataset_config)
    if args.type == "bpe":
        tokenizer = ReferenceBPETokenizer.train(documents, vocab_size=args.vocab_size)
        tokenizer.save_pretrained(
            Path(args.output),
            metadata={"training_corpus_sha256": corpus_sha256(documents), "tokenizer_algorithm": "byte_bpe"},
        )
        return

    layout = compute_vocab_layout(args.vocab_size, special_token_count=1, tail_token_count=256)
    if args.type == "memoryless" or args.algorithm == "memoryless":
        trie = build_memoryless_tree(b"\n".join(documents), layout.phrase_leaf_count, max_depth=args.max_depth)
        algorithm = "memoryless"
    else:
        trie = build_corpus_count_tree(documents, layout.phrase_leaf_count, args.expansion_batch_size, args.max_depth)
        algorithm = "corpus_count_batched"
    tokenizer = PrefixTreeTokenizer.from_trie(trie, vocab_size=args.vocab_size)
    save_tokenizer(
        tokenizer,
        Path(args.output),
        metadata={"training_corpus_sha256": corpus_sha256(documents), "tree_algorithm": algorithm},
    )


if __name__ == "__main__":
    main()
