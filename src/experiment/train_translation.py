from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import time

from .transformer_translation import TransformerConfig, build_model, paper_learning_rate
from .translation_data import IndexedParallelDataset, collate_pairs, token_batches


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Transformer-base model from Attention Is All You Need.")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--tokens-per-batch", type=int, default=25000, help="Approximate source and target tokens globally.")
    parser.add_argument(
        "--microbatch-tokens",
        type=int,
        default=1024,
        help="Maximum source and target tokens materialized at once; gradients accumulate to --tokens-per-batch.",
    )
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--validate-steps", type=int, default=1000)
    parser.add_argument("--validation-batches", type=int, default=100)
    parser.add_argument(
        "--validation-tokens-per-batch",
        type=int,
        default=2048,
        help="Validation-only memory budget; does not affect training batches or metrics.",
    )
    parser.add_argument("--log-steps", type=int, default=100)
    parser.add_argument("--keep-checkpoints", type=int, default=5)
    parser.add_argument("--resume", help="Checkpoint path, or 'auto' for the latest output checkpoint.")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision on CUDA.")
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    train(args)


def train(args: argparse.Namespace) -> None:
    try:
        import torch
        import torch.distributed as dist
        import torch.nn.functional as functional
        from torch.nn.parallel import DistributedDataParallel
        from tqdm.auto import tqdm
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("train_translation requires PyTorch 2.0 or newer") from exc

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_data = IndexedParallelDataset(args.train_data)
    if len(train_data) == 0:
        raise ValueError("training dataset is empty")
    validation_data = IndexedParallelDataset(args.validation_data) if args.validation_data else None
    metadata = train_data.metadata
    config = TransformerConfig(
        vocab_size=metadata["vocab_size"],
        d_model=args.d_model,
        d_ff=args.d_ff,
        heads=args.heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        max_length=args.max_length,
        pad_token_id=metadata["pad_token_id"],
    )
    model = build_model(config).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    start_step = _resume_if_requested(args.resume, Path(args.output), model, optimizer, device)
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        run_metadata = {
            "paper": "Attention Is All You Need (Vaswani et al., 2017)",
            "dataset": "WMT 2014 English-German",
            "model": config.to_dict(),
            "optimizer": {"name": "Adam", "betas": [0.9, 0.98], "epsilon": 1e-9},
            "schedule": {"name": "paper_inverse_sqrt", "warmup_steps": args.warmup_steps},
            "label_smoothing": args.label_smoothing,
            "tokens_per_batch": args.tokens_per_batch,
            "microbatch_tokens": args.microbatch_tokens,
            "steps": args.steps,
            "seed": args.seed,
            "world_size": world_size,
            "train_data": str(args.train_data),
            "validation_data": args.validation_data,
            "parameters": sum(parameter.numel() for parameter in _unwrap(model).parameters()),
        }
        (output / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8")

    local_budget = max(1, args.tokens_per_batch // world_size)
    microbatch_budget = min(local_budget, args.microbatch_tokens)
    batches = token_batches(train_data, microbatch_budget, seed=args.seed, rank=rank, world_size=world_size)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    autocast = (lambda: torch.autocast("cuda", dtype=torch.float16)) if scaler.is_enabled() else nullcontext
    running_loss = running_tokens = 0.0
    started = time.monotonic()
    model.train()
    progress = tqdm(
        range(start_step + 1, args.steps + 1),
        total=args.steps,
        initial=start_step,
        desc="Training Transformer",
        unit="step",
        disable=rank != 0,
    )
    for step in progress:
        learning_rate = paper_learning_rate(step, config.d_model, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_tokens = 0
        microbatches = 0
        while step_tokens < local_budget:
            source, target_input, target_output = collate_pairs(train_data, next(batches), device)
            predicted_tokens = target_output.ne(config.pad_token_id).sum().item()
            sync_context = (
                model.no_sync()
                if distributed and step_tokens + predicted_tokens < local_budget
                else nullcontext()
            )
            with sync_context:
                with autocast():
                    logits = model(source, target_input)
                    loss_sum = functional.cross_entropy(
                        logits.reshape(-1, config.vocab_size),
                        target_output.reshape(-1),
                        ignore_index=config.pad_token_id,
                        label_smoothing=args.label_smoothing,
                        reduction="sum",
                    )
                    normalized_loss = loss_sum / local_budget
                scaler.scale(normalized_loss).backward()
            step_loss += float(loss_sum.detach())
            step_tokens += predicted_tokens
            microbatches += 1
            del source, target_input, target_output, logits, loss_sum, normalized_loss
        scaler.step(optimizer)
        scaler.update()
        running_loss += step_loss
        running_tokens += step_tokens
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_steps == 0:
            totals = torch.tensor([running_loss, running_tokens], dtype=torch.float64, device=device)
            if distributed:
                dist.all_reduce(totals)
            if rank == 0:
                elapsed = time.monotonic() - started
                metrics = {
                    "step": step,
                    "loss": totals[0].item() / totals[1].item(),
                    "learning_rate": learning_rate,
                    "target_tokens_per_second": totals[1].item() / elapsed,
                }
                progress.set_postfix(
                    loss=f"{metrics['loss']:.4f}",
                    lr=f"{learning_rate:.2e}",
                    microbatches=microbatches,
                )
                tqdm.write(json.dumps(metrics))
            running_loss = running_tokens = 0.0
            started = time.monotonic()

        if validation_data is not None and args.validate_steps and step % args.validate_steps == 0:
            validation_loss = evaluate_loss(
                model,
                validation_data,
                device,
                config,
                token_budget=min(local_budget, args.validation_tokens_per_batch),
                batches=args.validation_batches,
                rank=rank,
                world_size=world_size,
            )
            if distributed:
                dist.all_reduce(validation_loss)
            if rank == 0:
                tqdm.write(json.dumps({"step": step, "validation_loss": (validation_loss[0] / validation_loss[1]).item()}))

        if args.save_steps and step % args.save_steps == 0:
            if rank == 0:
                _save_checkpoint(output, step, model, optimizer, config, args.keep_checkpoints)
            if distributed:
                dist.barrier()

    if rank == 0 and (not args.save_steps or args.steps % args.save_steps):
        _save_checkpoint(output, args.steps, model, optimizer, config, args.keep_checkpoints)
    progress.close()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def evaluate_loss(model, dataset, device, config, *, token_budget: int, batches: int, rank: int, world_size: int):
    import torch
    import torch.nn.functional as functional

    was_training = model.training
    model.eval()
    total_loss = total_tokens = 0.0
    iterator = token_batches(dataset, token_budget, seed=0, rank=rank, world_size=world_size)
    with torch.inference_mode():
        for _ in range(batches):
            source, target_input, target_output = collate_pairs(dataset, next(iterator), device)
            logits = model(source, target_input)
            loss = functional.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                target_output.reshape(-1),
                ignore_index=config.pad_token_id,
            )
            count = target_output.ne(config.pad_token_id).sum().item()
            total_loss += float(loss) * count
            total_tokens += count
    model.train(was_training)
    return torch.tensor([total_loss, total_tokens], dtype=torch.float64, device=device)


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _save_checkpoint(output: Path, step: int, model, optimizer, config: TransformerConfig, keep: int) -> None:
    import torch

    path = output / f"checkpoint-{step:06d}.pt"
    torch.save(
        {"step": step, "model": _unwrap(model).state_dict(), "optimizer": optimizer.state_dict(), "config": config.to_dict()},
        path,
    )
    checkpoints = sorted(output.glob("checkpoint-*.pt"))
    for old_path in checkpoints[:-keep] if keep > 0 else checkpoints[:-1]:
        old_path.unlink()


def _resume_if_requested(resume: str | None, output: Path, model, optimizer, device) -> int:
    if not resume:
        return 0
    if resume == "auto":
        checkpoints = sorted(output.glob("checkpoint-*.pt"))
        if not checkpoints:
            return 0
        path = checkpoints[-1]
    else:
        path = Path(resume)
    import torch

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    _unwrap(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["step"])


if __name__ == "__main__":
    main()
