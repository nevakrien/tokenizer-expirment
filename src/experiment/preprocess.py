from __future__ import annotations

import argparse
import json
from pathlib import Path

from prefix_tokenizer import PrefixTreeTokenizer
from prefix_tokenizer.framing import encode_documents, pack_token_blocks

from .common import load_documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--dataset-config")
    parser.add_argument("--input-files", nargs="*")
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tokenizer = PrefixTreeTokenizer.from_pretrained(args.tokenizer)
    documents = load_documents(args.input_files, args.dataset_config)
    encoded = encode_documents(tokenizer, documents)
    blocks = pack_token_blocks(encoded, args.context_length)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "data.jsonl").write_text("".join(json.dumps({"input_ids": block, "attention_mask": [1] * len(block)}) + "\n" for block in blocks), encoding="utf-8")
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "document_count": len(documents),
                "block_count": len(blocks),
                "context_length": args.context_length,
                "tokenizer": args.tokenizer,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
