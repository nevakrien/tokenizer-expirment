import json

import pytest

from experiment.train_clm import CLMExample, load_examples, split_train_validation


def test_load_examples_accepts_preprocess_directory(tmp_path) -> None:
    data = tmp_path / "data.jsonl"
    data.write_text(
        json.dumps({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}) + "\n",
        encoding="utf-8",
    )

    assert load_examples(tmp_path, context_length=3) == [CLMExample(input_ids=[1, 2, 3])]


def test_load_examples_rejects_wrong_context_length(tmp_path) -> None:
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"input_ids": [1, 2]}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected 3 tokens"):
        load_examples(tmp_path, context_length=3)


def test_split_train_validation_is_seeded_and_non_empty() -> None:
    examples = [CLMExample(input_ids=[index]) for index in range(10)]

    train_a, validation_a = split_train_validation(examples, validation_fraction=0.2, seed=7)
    train_b, validation_b = split_train_validation(examples, validation_fraction=0.2, seed=7)

    assert train_a == train_b
    assert validation_a == validation_b
    assert len(train_a) == 8
    assert len(validation_a) == 2


def test_split_train_validation_can_disable_validation() -> None:
    examples = [CLMExample(input_ids=[1])]

    assert split_train_validation(examples, validation_fraction=0, seed=0) == (examples, [])
