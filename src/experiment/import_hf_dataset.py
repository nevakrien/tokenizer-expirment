from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a Hugging Face dataset split into local JSONL text documents.")
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--name", default="wikitext-103-raw-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--output", required=True, help="Output .jsonl path.")
    parser.add_argument("--max-docs", type=int, default=10000)
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument("--streaming", action="store_true", help="Stream examples instead of downloading the full split first.")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("import_hf_dataset requires the 'datasets' package") from exc

    dataset_kwargs: dict[str, Any] = {"split": args.split, "streaming": args.streaming}
    if args.name:
        dataset = load_dataset(args.dataset, args.name, **dataset_kwargs)
    else:
        dataset = load_dataset(args.dataset, **dataset_kwargs)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    total_chars = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in dataset:
            text = row.get(args.text_field) if isinstance(row, dict) else None
            if not isinstance(text, str):
                raise ValueError(f"dataset row does not contain string field {args.text_field!r}")
            text = text.strip()
            if len(text) < args.min_chars:
                continue
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
            total_chars += len(text)
            if args.max_docs > 0 and count >= args.max_docs:
                break

    metadata = {
        "dataset": args.dataset,
        "name": args.name,
        "split": args.split,
        "text_field": args.text_field,
        "documents": count,
        "characters": total_chars,
        "output": str(output),
    }
    output.with_suffix(output.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
