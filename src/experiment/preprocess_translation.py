from __future__ import annotations

import argparse
import json

from .translation_data import preprocess_parallel


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode aligned translation data into an indexed binary dataset.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=256, help="Drop pairs exceeding this token length; zero disables.")
    args = parser.parse_args()
    metadata = preprocess_parallel(args.source, args.target, args.tokenizer, args.output, max_length=args.max_length)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
