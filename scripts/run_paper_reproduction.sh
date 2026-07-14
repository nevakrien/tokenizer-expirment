#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"

DATASET="${DATASET:-Salesforce/wikitext}"
DATASET_NAME="${DATASET_NAME:-wikitext-103-raw-v1}"
MAX_DOCS="${MAX_DOCS:-10000}"
MIN_CHARS="${MIN_CHARS:-20}"
SEEDS="${SEEDS:-0 1 2}"

usage() {
  printf '%s\n' \
    "usage: $0 COMMAND" \
    "" \
    "Commands:" \
    "  smoke                    Tiny local smoke run." \
    "  import-wikitext           Import WikiText train and validation splits." \
    "  small-clm-wikitext        8K-vocab controlled BPE vs prefix CLM comparison." \
    "  gpt1-tokenizers           GPT-1-style 40K tokenizer-only comparison." \
    "  gpt2-small-tokenizers     GPT-2-small-style 50257 tokenizer-only comparison." \
    "" \
    "Useful environment variables:" \
    "  MAX_DOCS=0 for full split import, default 10000" \
    "  SEEDS='0 1 2', default three seeds" \
    "  MAX_STEPS=10000, default for small-clm-wikitext" \
    "  BATCH_SIZE=8, GRAD_ACCUM=1"
}

import_wikitext() {
  python -m experiment.import_hf_dataset \
    --dataset "${DATASET}" \
    --name "${DATASET_NAME}" \
    --split train \
    --output data/imported/wikitext103/train.jsonl \
    --max-docs "${MAX_DOCS}" \
    --min-chars "${MIN_CHARS}"

  python -m experiment.import_hf_dataset \
    --dataset "${DATASET}" \
    --name "${DATASET_NAME}" \
    --split validation \
    --output data/imported/wikitext103/validation.jsonl \
    --max-docs "${MAX_DOCS}" \
    --min-chars "${MIN_CHARS}"
}

train_tokenizer_pair() {
  local vocab_size="$1"
  local stem="$2"

  python -m experiment.train_tokenizer \
    --type bpe \
    --dataset-config configs/data/wikitext103_train_subset.json \
    --vocab-size "${vocab_size}" \
    --output "artifacts/tokenizers/${stem}_bpe_${vocab_size}"

  python -m experiment.train_tokenizer \
    --type prefix_tree \
    --algorithm corpus_count_batched \
    --dataset-config configs/data/wikitext103_train_subset.json \
    --vocab-size "${vocab_size}" \
    --output "artifacts/tokenizers/${stem}_prefix_${vocab_size}"

  python -m experiment.verify_tokenizer \
    "artifacts/tokenizers/${stem}_prefix_${vocab_size}" \
    --random-cases "${VERIFY_RANDOM_CASES:-100000}"

  python -m experiment.analyze_tokenizers \
    --tokenizers \
    "artifacts/tokenizers/${stem}_bpe_${vocab_size}" \
    "artifacts/tokenizers/${stem}_prefix_${vocab_size}" \
    --dataset-config configs/data/wikitext103_validation_subset.json \
    --output "reports/${stem}_tokenizers_${vocab_size}.json"
}

preprocess_pair() {
  local vocab_size="$1"
  local stem="$2"
  local context_length="$3"

  python -m experiment.preprocess \
    --tokenizer "artifacts/tokenizers/${stem}_bpe_${vocab_size}" \
    --dataset-config configs/data/wikitext103_train_subset.json \
    --context-length "${context_length}" \
    --output "artifacts/blocks/${stem}_bpe_${vocab_size}_ctx${context_length}"

  python -m experiment.preprocess \
    --tokenizer "artifacts/tokenizers/${stem}_prefix_${vocab_size}" \
    --dataset-config configs/data/wikitext103_train_subset.json \
    --context-length "${context_length}" \
    --output "artifacts/blocks/${stem}_prefix_${vocab_size}_ctx${context_length}"
}

train_and_evaluate_pair() {
  local vocab_size="$1"
  local stem="$2"
  local context_length="$3"
  local layers="$4"
  local hidden_size="$5"
  local heads="$6"
  local ffn_size="$7"
  local max_steps="${MAX_STEPS:-10000}"

  for seed in ${SEEDS}; do
    for tokenizer_type in bpe prefix; do
      python -m experiment.train_clm \
        --train-data "artifacts/blocks/${stem}_${tokenizer_type}_${vocab_size}_ctx${context_length}" \
        --output "artifacts/clm/${stem}_${tokenizer_type}_${vocab_size}_ctx${context_length}_seed${seed}" \
        --vocab-size "${vocab_size}" \
        --context-length "${context_length}" \
        --layers "${layers}" \
        --hidden-size "${hidden_size}" \
        --heads "${heads}" \
        --ffn-size "${ffn_size}" \
        --seed "${seed}" \
        --max-steps "${max_steps}" \
        --per-device-train-batch-size "${BATCH_SIZE:-8}" \
        --gradient-accumulation-steps "${GRAD_ACCUM:-1}" \
        --logging-steps "${LOGGING_STEPS:-100}" \
        --eval-steps "${EVAL_STEPS:-500}" \
        --save-steps "${SAVE_STEPS:-1000}"

      python -m experiment.evaluate_clm \
        --model "artifacts/clm/${stem}_${tokenizer_type}_${vocab_size}_ctx${context_length}_seed${seed}" \
        --tokenizer "artifacts/tokenizers/${stem}_${tokenizer_type}_${vocab_size}" \
        --dataset-config configs/data/wikitext103_validation_subset.json \
        --context-length "${context_length}" \
        --output "reports/${stem}_${tokenizer_type}_${vocab_size}_ctx${context_length}_seed${seed}_eval.json"
    done
  done
}

case "${1:-}" in
  smoke)
    python -m experiment.train_tokenizer --type prefix_tree --dataset-config configs/data/sample.json --vocab-size 1025 --output artifacts/tokenizers/sample_prefix
    python -m experiment.train_tokenizer --type bpe --dataset-config configs/data/sample.json --vocab-size 1025 --output artifacts/tokenizers/sample_bpe
    python -m experiment.analyze_tokenizers --tokenizers artifacts/tokenizers/sample_prefix artifacts/tokenizers/sample_bpe --dataset-config configs/data/sample.json --output reports/sample_tokenization.json
    python -m experiment.preprocess --tokenizer artifacts/tokenizers/sample_bpe --dataset-config configs/data/sample.json --context-length 8 --output artifacts/blocks/sample_bpe
    python -m experiment.preprocess --tokenizer artifacts/tokenizers/sample_prefix --dataset-config configs/data/sample.json --context-length 8 --output artifacts/blocks/sample_prefix
    ;;
  import-wikitext)
    import_wikitext
    ;;
  small-clm-wikitext)
    import_wikitext
    train_tokenizer_pair 8192 wikitext103
    preprocess_pair 8192 wikitext103 512
    train_and_evaluate_pair 8192 wikitext103 512 6 384 6 1536
    ;;
  gpt1-tokenizers)
    import_wikitext
    train_tokenizer_pair 40000 wikitext103
    ;;
  gpt2-small-tokenizers)
    import_wikitext
    train_tokenizer_pair 50257 wikitext103
    ;;
  *)
    usage
    exit 2
    ;;
esac
