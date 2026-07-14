from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CLMExample:
    input_ids: list[int]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small causal LM on preprocessed token blocks.")
    parser.add_argument("--train-data", required=True, help="Path to a preprocess output directory or data.jsonl file.")
    parser.add_argument("--validation-data", help="Optional validation preprocess directory or data.jsonl file.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--ffn-size", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    args = parser.parse_args()

    try:
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, set_seed
    except ImportError as exc:  # pragma: no cover - depends on optional packages
        raise SystemExit("train_clm requires optional dependencies: install torch and transformers") from exc

    set_seed(args.seed)
    train_examples = load_examples(args.train_data, context_length=args.context_length)
    if args.validation_data:
        validation_examples = load_examples(args.validation_data, context_length=args.context_length)
    else:
        train_examples, validation_examples = split_train_validation(
            train_examples,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )

    config = GPT2Config(
        vocab_size=args.vocab_size,
        n_positions=args.context_length,
        n_ctx=args.context_length,
        n_embd=args.hidden_size,
        n_layer=args.layers,
        n_head=args.heads,
        n_inner=args.ffn_size,
        resid_pdrop=args.dropout,
        embd_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        bos_token_id=None,
        eos_token_id=None,
    )
    model = GPT2LMHeadModel(config)
    training_args = _training_arguments(args, has_validation=bool(validation_examples))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TokenBlockDataset(train_examples, torch),
        eval_dataset=TokenBlockDataset(validation_examples, torch) if validation_examples else None,
    )
    trainer.train()
    trainer.save_model(args.output)
    _write_run_metadata(args, train_examples, validation_examples)


def load_examples(path: str | Path, *, context_length: int) -> list[CLMExample]:
    data_path = _data_jsonl_path(path)
    examples: list[CLMExample] = []
    with data_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            input_ids = item.get("input_ids")
            if not isinstance(input_ids, list) or not all(isinstance(token_id, int) for token_id in input_ids):
                raise ValueError(f"{data_path}:{line_number}: input_ids must be a list of integers")
            if len(input_ids) != context_length:
                raise ValueError(f"{data_path}:{line_number}: expected {context_length} tokens, found {len(input_ids)}")
            examples.append(CLMExample(input_ids=input_ids))
    if not examples:
        raise ValueError(f"no training examples found in {data_path}")
    return examples


def split_train_validation(
    examples: list[CLMExample],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[CLMExample], list[CLMExample]]:
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if not examples:
        raise ValueError("cannot split an empty dataset")
    if validation_fraction == 0 or len(examples) == 1:
        return examples, []
    validation_count = max(1, int(round(len(examples) * validation_fraction)))
    validation_count = min(validation_count, len(examples) - 1)
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    return shuffled[validation_count:], shuffled[:validation_count]


class TokenBlockDataset:
    def __init__(self, examples: list[CLMExample], torch_module) -> None:
        self.examples = examples
        self.torch = torch_module

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        input_ids = self.torch.tensor(self.examples[index].input_ids, dtype=self.torch.long)
        return {"input_ids": input_ids, "attention_mask": self.torch.ones_like(input_ids), "labels": input_ids.clone()}


def _data_jsonl_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        path = path / "data.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _training_arguments(args: argparse.Namespace, *, has_validation: bool):
    from transformers import TrainingArguments

    kwargs = {
        "output_dir": args.output,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "report_to": [],
        "seed": args.seed,
    }
    if has_validation:
        kwargs.update({"eval_steps": args.eval_steps, "load_best_model_at_end": False})
        try:
            return TrainingArguments(**kwargs, eval_strategy="steps")
        except TypeError:  # pragma: no cover - older Transformers uses evaluation_strategy
            return TrainingArguments(**kwargs, evaluation_strategy="steps")
    return TrainingArguments(**kwargs)


def _write_run_metadata(args: argparse.Namespace, train_examples: list[CLMExample], validation_examples: list[CLMExample]) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "layers": args.layers,
        "hidden_size": args.hidden_size,
        "heads": args.heads,
        "ffn_size": args.ffn_size,
        "dropout": args.dropout,
        "seed": args.seed,
    }
    (output / "clm_run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
