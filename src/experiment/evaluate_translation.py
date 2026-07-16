from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .reference_bpe import load_any_tokenizer
from .transformer_translation import TransformerConfig, build_model
from .translation_data import count_jsonl_records, iter_jsonl_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate WMT 2014 En-De with paper-style beam search and BLEU.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--average-checkpoints", type=int, default=5)
    parser.add_argument("--max-sentences", type=int, default=0)
    parser.add_argument("--device")
    args = parser.parse_args()
    evaluate(args)


def evaluate(args: argparse.Namespace) -> None:
    try:
        import sacrebleu
        import torch
        from tqdm.auto import tqdm
    except ImportError as exc:  # pragma: no cover - optional dependencies
        raise SystemExit("evaluate_translation requires torch and sacrebleu") from exc

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoints = sorted(Path(args.run).glob("checkpoint-*.pt"))[-args.average_checkpoints :]
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found in {args.run}")
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in checkpoints]
    config = TransformerConfig(**payloads[-1]["config"])
    model = build_model(config)
    averaged = {}
    for name, value in payloads[-1]["model"].items():
        if value.is_floating_point():
            averaged[name] = sum(payload["model"][name].float() for payload in payloads) / len(payloads)
            averaged[name] = averaged[name].to(value.dtype)
        else:
            averaged[name] = value
    model.load_state_dict(averaged)
    model.to(device).eval()
    tokenizer = load_any_tokenizer(args.tokenizer)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("tokenizer and model vocabulary sizes differ")

    hypotheses = []
    references = []
    source_iter = iter_jsonl_text(args.source)
    reference_iter = iter_jsonl_text(args.reference)
    total_sentences = count_jsonl_records(args.source)
    if args.max_sentences:
        total_sentences = min(total_sentences, args.max_sentences)
    pairs = tqdm(
        zip(source_iter, reference_iter, strict=True),
        total=total_sentences,
        desc="Translating newstest2014",
        unit="sentence",
    )
    for index, (source, reference) in enumerate(pairs):
        if args.max_sentences and index >= args.max_sentences:
            break
        source_ids = tokenizer.encode(source, add_special_tokens=False) + [tokenizer.eos_token_id]
        generated = beam_search(
            model,
            source_ids,
            tokenizer,
            device,
            beam_size=args.beam_size,
            alpha=args.length_penalty,
            max_length=min(len(source_ids) + 50, config.max_length),
        )
        hypotheses.append(tokenizer.decode_bytes(generated, skip_special_tokens=True).decode("utf-8", errors="replace"))
        references.append(reference)

    bleu_metric = sacrebleu.metrics.BLEU()
    bleu = bleu_metric.corpus_score(hypotheses, [references])
    report = {
        "paper": "Attention Is All You Need (Vaswani et al., 2017)",
        "dataset": "WMT 2014 English-German newstest2014",
        "sentences": len(hypotheses),
        "bleu": bleu.score,
        "sacrebleu_signature": str(bleu_metric.get_signature()),
        "beam_size": args.beam_size,
        "length_penalty": args.length_penalty,
        "checkpoints": [str(path) for path in checkpoints],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    output.with_suffix(".hypotheses.txt").write_text("\n".join(hypotheses) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def beam_search(model, source_ids, tokenizer, device, *, beam_size: int, alpha: float, max_length: int) -> list[int]:
    import torch
    import torch.nn.functional as functional

    source = torch.tensor([source_ids], dtype=torch.long, device=device)
    beams = [([tokenizer.bos_token_id], 0.0, False)]
    blocked = set(tokenizer.reserved_ids or set()) | {tokenizer.bos_token_id, tokenizer.pad_token_id}
    with torch.inference_mode():
        for _ in range(max_length):
            candidates = []
            for tokens, score, finished in beams:
                if finished:
                    candidates.append((tokens, score, True))
                    continue
                target = torch.tensor([tokens], dtype=torch.long, device=device)
                logits = model(source, target)[0, -1]
                if blocked:
                    logits[list(blocked)] = -torch.inf
                values, indices = functional.log_softmax(logits, dim=-1).topk(beam_size)
                for value, token_id in zip(values.tolist(), indices.tolist(), strict=True):
                    candidates.append((tokens + [token_id], score + value, token_id == tokenizer.eos_token_id))
            candidates.sort(key=lambda item: item[1] / _length_penalty(len(item[0]) - 1, alpha), reverse=True)
            beams = candidates[:beam_size]
            if all(finished for _, _, finished in beams):
                break
    best = max(beams, key=lambda item: item[1] / _length_penalty(len(item[0]) - 1, alpha))[0]
    return best[1:]


def _length_penalty(length: int, alpha: float) -> float:
    return ((5.0 + max(1, length)) / 6.0) ** alpha


if __name__ == "__main__":
    main()
