from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the WMT 2014 English-German paper dataset.")
    parser.add_argument("--output", default="data/imported/wmt14")
    parser.add_argument("--dataset", default="wmt/wmt14")
    parser.add_argument("--name", default="de-en")
    parser.add_argument("--max-pairs", type=int, default=0, help="Debug-only limit; zero imports each complete split.")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from tqdm.auto import tqdm
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("import_wmt14 requires the 'datasets' package") from exc

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    split_metadata = {}
    for split in ("train", "validation", "test"):
        dataset = load_dataset(args.dataset, args.name, split=split, streaming=args.streaming)
        split_info = dataset.info.splits.get(split) if dataset.info.splits else None
        total = split_info.num_examples if split_info is not None else None
        if args.max_pairs and total is not None:
            total = min(total, args.max_pairs)
        paths = {language: output / f"{split}.{language}.jsonl" for language in ("en", "de")}
        digest = hashlib.sha256()
        count = 0
        with paths["en"].open("w", encoding="utf-8") as en_handle, paths["de"].open(
            "w", encoding="utf-8"
        ) as de_handle:
            for row in tqdm(dataset, total=total, desc=f"Importing WMT14 {split}", unit="pair"):
                translation = row.get("translation")
                if not isinstance(translation, dict) or not all(isinstance(translation.get(lang), str) for lang in paths):
                    raise ValueError(f"{split} row {count + 1} has no en/de translation pair")
                en = translation["en"].strip()
                de = translation["de"].strip()
                en_handle.write(json.dumps({"text": en}, ensure_ascii=False) + "\n")
                de_handle.write(json.dumps({"text": de}, ensure_ascii=False) + "\n")
                for text in (en, de):
                    encoded = text.encode("utf-8")
                    digest.update(len(encoded).to_bytes(8, "little"))
                    digest.update(encoded)
                count += 1
                if args.max_pairs and count >= args.max_pairs:
                    break
        split_metadata[split] = {"pairs": count, "sha256": digest.hexdigest()}

    metadata = {
        "dataset": args.dataset,
        "name": args.name,
        "source_language": "en",
        "target_language": "de",
        "debug_limited": bool(args.max_pairs),
        "max_pairs": args.max_pairs,
        "splits": split_metadata,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
