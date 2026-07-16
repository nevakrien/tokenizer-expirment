from __future__ import annotations

import json
import mmap
from pathlib import Path
import random
import struct
from typing import Iterator

from .reference_bpe import load_any_tokenizer


INDEX = struct.Struct("<QIQI")


def iter_jsonl_text(path: str | Path) -> Iterator[str]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise ValueError(f"{path}:{line_number}: expected an object with a string 'text' field")
            yield item["text"]


def count_jsonl_records(path: str | Path) -> int:
    with Path(path).open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def preprocess_parallel(
    source_path: str | Path,
    target_path: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
    *,
    max_length: int = 256,
) -> dict:
    from tqdm.auto import tqdm

    tokenizer = load_any_tokenizer(tokenizer_path)
    require_translation_specials(tokenizer)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    source_offset = target_offset = kept = dropped = 0
    source_iter = iter_jsonl_text(source_path)
    target_iter = iter_jsonl_text(target_path)
    sentinel = object()
    with (output / "source.bin").open("wb") as source_bin, (output / "target.bin").open("wb") as target_bin, (
        output / "pairs.idx"
    ).open("wb") as index:
        progress = tqdm(
            total=count_jsonl_records(source_path),
            desc=f"Encoding {Path(source_path).stem}",
            unit="pair",
        )
        while True:
            source = next(source_iter, sentinel)
            target = next(target_iter, sentinel)
            if source is sentinel and target is sentinel:
                break
            if source is sentinel or target is sentinel:
                raise ValueError("source and target files contain different numbers of records")
            source_ids = tokenizer.encode(source, add_special_tokens=False) + [tokenizer.eos_token_id]
            target_ids = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
            if max_length and (len(source_ids) > max_length or len(target_ids) > max_length):
                dropped += 1
                progress.update(1)
                continue
            source_bin.write(struct.pack(f"<{len(source_ids)}I", *source_ids))
            target_bin.write(struct.pack(f"<{len(target_ids)}I", *target_ids))
            index.write(INDEX.pack(source_offset, len(source_ids), target_offset, len(target_ids)))
            source_offset += len(source_ids)
            target_offset += len(target_ids)
            kept += 1
            progress.update(1)
        progress.close()
    metadata = {
        "source": str(source_path),
        "target": str(target_path),
        "tokenizer": str(tokenizer_path),
        "vocab_size": tokenizer.vocab_size,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "pairs": kept,
        "dropped_pairs": dropped,
        "max_length": max_length,
        "source_tokens": source_offset,
        "target_tokens": target_offset,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


class IndexedParallelDataset:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.metadata = json.loads((self.path / "metadata.json").read_text(encoding="utf-8"))
        self._files = []
        self._maps = []
        for name in ("pairs.idx", "source.bin", "target.bin"):
            handle = (self.path / name).open("rb")
            self._files.append(handle)
            self._maps.append(mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ))
        self.index, self.source, self.target = self._maps
        if len(self.index) % INDEX.size:
            raise ValueError(f"corrupt index: {self.path / 'pairs.idx'}")

    def __len__(self) -> int:
        return len(self.index) // INDEX.size

    def lengths(self, index: int) -> tuple[int, int]:
        _, source_length, _, target_length = INDEX.unpack_from(self.index, index * INDEX.size)
        return source_length, target_length

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        source_offset, source_length, target_offset, target_length = INDEX.unpack_from(self.index, index * INDEX.size)
        source = list(struct.unpack_from(f"<{source_length}I", self.source, source_offset * 4))
        target = list(struct.unpack_from(f"<{target_length}I", self.target, target_offset * 4))
        return source, target


def token_batches(
    dataset: IndexedParallelDataset,
    token_budget: int,
    *,
    seed: int,
    rank: int = 0,
    world_size: int = 1,
    shuffle_buffer: int = 8192,
) -> Iterator[list[int]]:
    if token_budget <= 0:
        raise ValueError("token budget must be positive")
    epoch = 0
    while True:
        rng = random.Random(seed + epoch)
        buffer: list[int] = []
        for index in range(rank, len(dataset), world_size):
            buffer.append(index)
            if len(buffer) == shuffle_buffer:
                yield from _batches_from_buffer(dataset, buffer, token_budget, rng)
                buffer = []
        if buffer:
            yield from _batches_from_buffer(dataset, buffer, token_budget, rng)
        epoch += 1


def _batches_from_buffer(
    dataset: IndexedParallelDataset, indices: list[int], token_budget: int, rng: random.Random
) -> Iterator[list[int]]:
    rng.shuffle(indices)
    indices.sort(key=lambda index: max(dataset.lengths(index)))
    batches: list[list[int]] = []
    batch: list[int] = []
    max_source = max_target = 0
    for index in indices:
        source_length, target_length = dataset.lengths(index)
        next_source = max(max_source, source_length)
        next_target = max(max_target, target_length)
        if batch and (next_source * (len(batch) + 1) > token_budget or next_target * (len(batch) + 1) > token_budget):
            batches.append(batch)
            batch = []
            max_source = max_target = 0
        batch.append(index)
        max_source = max(max_source, source_length)
        max_target = max(max_target, target_length)
    if batch:
        batches.append(batch)
    rng.shuffle(batches)
    yield from batches


def collate_pairs(dataset: IndexedParallelDataset, indices: list[int], device):
    import torch

    examples = [dataset[index] for index in indices]
    pad = dataset.metadata["pad_token_id"]
    bos = dataset.metadata["bos_token_id"]
    source_length = max(len(source) for source, _ in examples)
    target_length = max(len(target) for _, target in examples)
    source_batch = torch.full((len(examples), source_length), pad, dtype=torch.long, device=device)
    target_input = torch.full((len(examples), target_length), pad, dtype=torch.long, device=device)
    target_output = torch.full((len(examples), target_length), pad, dtype=torch.long, device=device)
    for row, (source, target) in enumerate(examples):
        source_batch[row, : len(source)] = torch.tensor(source, device=device)
        target_input[row, 0] = bos
        if len(target) > 1:
            target_input[row, 1 : len(target)] = torch.tensor(target[:-1], device=device)
        target_output[row, : len(target)] = torch.tensor(target, device=device)
    return source_batch, target_input, target_output


def require_translation_specials(tokenizer) -> None:
    if tokenizer.bos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("translation requires a tokenizer trained with --translation-special-tokens")
    if len({tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id}) != 3:
        raise ValueError("EOS, BOS, and PAD IDs must be distinct")
