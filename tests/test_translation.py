import json
from types import SimpleNamespace

import pytest

from experiment.reference_bpe import ReferenceBPETokenizer
from experiment.transformer_translation import TransformerConfig, build_model, paper_learning_rate
from experiment.translation_data import IndexedParallelDataset, collate_pairs, count_jsonl_records, preprocess_parallel, token_batches
from experiment.train_translation import train


def test_paper_schedule_warms_up_then_decays() -> None:
    assert paper_learning_rate(1) < paper_learning_rate(4000)
    assert paper_learning_rate(4001) < paper_learning_rate(4000)


def test_indexed_parallel_preprocessing_and_batching(tmp_path) -> None:
    pytest.importorskip("tokenizers")
    source = tmp_path / "train.en.jsonl"
    target = tmp_path / "train.de.jsonl"
    source.write_text('\n'.join(json.dumps({"text": text}) for text in ["hello", "good day"]) + '\n')
    target.write_text('\n'.join(json.dumps({"text": text}) for text in ["hallo", "guten tag"]) + '\n')
    assert count_jsonl_records(source) == 2
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer = ReferenceBPETokenizer.train([b"hello", b"good day", b"hallo", b"guten tag"], 300, special_token_count=3)
    tokenizer.save_pretrained(tokenizer_path)

    preprocess_parallel(source, target, tokenizer_path, tmp_path / "encoded", max_length=32)
    dataset = IndexedParallelDataset(tmp_path / "encoded")
    assert len(dataset) == 2
    batch = next(token_batches(dataset, 100, seed=1))
    source_ids, target_input, target_output = collate_pairs(dataset, batch, "cpu")
    assert source_ids.shape[0] == 2
    assert target_input[0, 0].item() == tokenizer.bos_token_id
    assert tokenizer.eos_token_id in target_output[0].tolist()


def test_tiny_transformer_forward() -> None:
    torch = pytest.importorskip("torch")
    config = TransformerConfig(vocab_size=32, d_model=16, d_ff=32, heads=4, encoder_layers=1, decoder_layers=1, max_length=8, pad_token_id=31)
    model = build_model(config)
    logits = model(torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5]]))
    assert logits.shape == (1, 2, 32)
    assert model.output.weight is model.embedding.weight


def test_tiny_training_run_writes_resumable_checkpoint(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("tokenizers")
    source = tmp_path / "train.en.jsonl"
    target = tmp_path / "train.de.jsonl"
    source.write_text(json.dumps({"text": "hello world"}) + "\n")
    target.write_text(json.dumps({"text": "hallo welt"}) + "\n")
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer = ReferenceBPETokenizer.train([b"hello world", b"hallo welt"], 300, special_token_count=3)
    tokenizer.save_pretrained(tokenizer_path)
    data_path = tmp_path / "encoded"
    preprocess_parallel(source, target, tokenizer_path, data_path, max_length=32)
    output = tmp_path / "run"

    train(SimpleNamespace(
        train_data=str(data_path), validation_data=None, output=str(output), steps=1,
        tokens_per_batch=32, microbatch_tokens=16, warmup_steps=4, label_smoothing=0.1, seed=1,
        save_steps=1, validate_steps=0, validation_batches=1, validation_tokens_per_batch=32, log_steps=1,
        keep_checkpoints=5, resume=None, amp=False, d_model=16, d_ff=32, heads=4,
        encoder_layers=1, decoder_layers=1, dropout=0.1, max_length=32,
    ))

    assert (output / "checkpoint-000001.pt").exists()
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["steps"] == 1
