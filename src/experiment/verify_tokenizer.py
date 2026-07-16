from __future__ import annotations

import argparse
import random

from tqdm.auto import tqdm

from prefix_tokenizer import PrefixTreeTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer")
    parser.add_argument("--random-cases", type=int, default=10_000)
    args = parser.parse_args()
    tokenizer = PrefixTreeTokenizer.from_pretrained(args.tokenizer)
    tokenizer.trie.validate(model_vocab_size=tokenizer.vocab_size, reserved_ids=tokenizer.reserved_ids)
    for value in tqdm(range(256), desc="Verifying one-byte inputs", unit="case"):
        data = bytes([value])
        assert tokenizer.decode_bytes(tokenizer.encode_bytes(data)) == data
    for value in tqdm(range(256 * 256), desc="Verifying two-byte inputs", unit="case"):
        data = bytes([value >> 8, value & 0xFF])
        assert tokenizer.decode_bytes(tokenizer.encode_bytes(data)) == data
    rng = random.Random(0)
    for _ in tqdm(range(args.random_cases), desc="Verifying random inputs", unit="case"):
        size = rng.randrange(0, 512)
        data = bytes(rng.randrange(256) for _ in range(size))
        assert tokenizer.decode_bytes(tokenizer.encode_bytes(data)) == data


if __name__ == "__main__":
    main()
