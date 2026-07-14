from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from prefix_tokenizer.framing import encode_documents, pack_token_blocks

from .common import load_documents
from .reference_bpe import load_any_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained causal LM on token blocks.")
    parser.add_argument("--model", "--run", dest="model", required=True, help="Path to a saved train_clm output directory.")
    parser.add_argument("--tokenizer", help="Tokenizer path. Required when evaluating from raw documents.")
    parser.add_argument("--dataset-config", help="Raw validation/test dataset config.")
    parser.add_argument("--input-files", nargs="*", help="Raw validation/test files.")
    parser.add_argument("--validation-data", help="Preprocessed validation directory or data.jsonl file.")
    parser.add_argument("--context-length", type=int, help="Required for raw documents; inferred for preprocessed data when possible.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        import torch
        from transformers import GPT2LMHeadModel
    except ImportError as exc:  # pragma: no cover - depends on optional packages
        raise SystemExit("evaluate_clm requires optional dependencies: install torch and transformers") from exc

    blocks, metadata = _load_evaluation_blocks(args)
    if not blocks:
        raise ValueError("no evaluation blocks found")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device)
    model.eval()

    total_nll_nats = 0.0
    total_predicted_tokens = 0
    with torch.no_grad():
        for start in range(0, len(blocks), args.batch_size):
            batch = blocks[start : start + args.batch_size]
            input_ids = torch.tensor(batch, dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=input_ids)
            predicted_tokens = input_ids.numel() - input_ids.shape[0]
            total_nll_nats += float(outputs.loss) * predicted_tokens
            total_predicted_tokens += predicted_tokens
    if total_predicted_tokens == 0:
        raise ValueError("evaluation requires context length greater than 1")

    loss_nats = total_nll_nats / total_predicted_tokens
    loss_bits = loss_nats / math.log(2)
    original_bytes = metadata.get("original_bytes")
    report = {
        "model": args.model,
        "block_count": len(blocks),
        "context_length": len(blocks[0]),
        "predicted_tokens": total_predicted_tokens,
        "loss_nats_per_token": loss_nats,
        "loss_bits_per_token": loss_bits,
        "perplexity_per_token": math.exp(loss_nats),
        "device": str(device),
        **metadata,
    }
    if original_bytes:
        report["bits_per_byte_approx"] = total_nll_nats / math.log(2) / original_bytes

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _load_evaluation_blocks(args: argparse.Namespace) -> tuple[list[list[int]], dict]:
    if args.validation_data:
        return _load_preprocessed_blocks(args.validation_data, args.context_length)

    if not args.tokenizer:
        raise ValueError("--tokenizer is required unless --validation-data is provided")
    if not args.context_length:
        raise ValueError("--context-length is required when evaluating raw documents")

    tokenizer = load_any_tokenizer(args.tokenizer)
    documents = load_documents(args.input_files, args.dataset_config)
    encoded = encode_documents(tokenizer, documents)
    total_tokens = sum(len(document) for document in encoded)
    blocks = pack_token_blocks(encoded, args.context_length)
    return blocks, {
        "dataset_config": args.dataset_config,
        "document_count": len(documents),
        "dropped_tokens": total_tokens - len(blocks) * args.context_length,
        "original_bytes": sum(len(document) for document in documents),
        "tokenizer": args.tokenizer,
        "total_tokens": total_tokens,
    }


def _load_preprocessed_blocks(path: str | Path, context_length: int | None) -> tuple[list[list[int]], dict]:
    path = Path(path)
    data_path = path / "data.jsonl" if path.is_dir() else path
    metadata_path = data_path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    expected_length = context_length or metadata.get("context_length")
    blocks: list[list[int]] = []
    with data_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            input_ids = item.get("input_ids")
            if not isinstance(input_ids, list) or not all(isinstance(token_id, int) for token_id in input_ids):
                raise ValueError(f"{data_path}:{line_number}: input_ids must be a list of integers")
            if expected_length is not None and len(input_ids) != expected_length:
                raise ValueError(f"{data_path}:{line_number}: expected {expected_length} tokens, found {len(input_ids)}")
            blocks.append(input_ids)
    return blocks, metadata


if __name__ == "__main__":
    main()
