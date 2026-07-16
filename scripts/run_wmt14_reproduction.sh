#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

ACTIVE_PID=""

interrupt() {
  trap - INT TERM
  printf '\nStopping reproduction...\n' >&2
  if [[ -n "${ACTIVE_PID}" ]] && kill -0 -- "-${ACTIVE_PID}" 2>/dev/null; then
    kill -INT -- "-${ACTIVE_PID}" 2>/dev/null || true
    for _ in {1..20}; do
      if ! kill -0 -- "-${ACTIVE_PID}" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
    if kill -0 -- "-${ACTIVE_PID}" 2>/dev/null; then
      kill -TERM -- "-${ACTIVE_PID}" 2>/dev/null || true
    fi
    wait "${ACTIVE_PID}" 2>/dev/null || true
  fi
  exit 130
}

trap interrupt INT TERM

run_command() {
  local status
  setsid "$@" &
  ACTIVE_PID=$!
  if wait "${ACTIVE_PID}"; then
    status=0
  else
    status=$?
  fi
  ACTIVE_PID=""
  return "${status}"
}

DATA_DIR="data/imported/wmt14"
VOCAB_SIZE=37000
STEPS=100000
TOKENS_PER_BATCH=25000
MAX_LENGTH=256
SEED=1
SAVE_STEPS=5000
NUM_GPUS="$(python -c 'import torch; print(torch.cuda.device_count())')"

if [[ ! -f "${DATA_DIR}/metadata.json" ]]; then
  run_command python -m experiment.import_wmt14 --output "${DATA_DIR}"
fi

prepare_tokenizer() {
  local kind="$1"
  local output="artifacts/tokenizers/wmt14_${kind}_${VOCAB_SIZE}"
  if [[ -f "${output}/metadata.json" ]]; then
    return
  fi
  if [[ "${kind}" == "prefix" ]]; then
    run_command python -m experiment.train_tokenizer \
      --type prefix_tree \
      --algorithm corpus_count_batched \
      --input-files "${DATA_DIR}/train.en.jsonl" "${DATA_DIR}/train.de.jsonl" \
      --vocab-size "${VOCAB_SIZE}" \
      --translation-special-tokens \
      --output "${output}"
    run_command python -m experiment.verify_tokenizer "${output}" --random-cases 100000
  else
    run_command python -m experiment.train_tokenizer \
      --type bpe \
      --input-files "${DATA_DIR}/train.en.jsonl" "${DATA_DIR}/train.de.jsonl" \
      --vocab-size "${VOCAB_SIZE}" \
      --translation-special-tokens \
      --output "${output}"
  fi
}

prepare_encoded_data() {
  local kind="$1"
  local tokenizer="artifacts/tokenizers/wmt14_${kind}_${VOCAB_SIZE}"
  for split in train validation; do
    local output="artifacts/translation/wmt14_${kind}_${VOCAB_SIZE}/${split}"
    if [[ -f "${output}/metadata.json" ]]; then
      continue
    fi
    run_command python -m experiment.preprocess_translation \
      --source "${DATA_DIR}/${split}.en.jsonl" \
      --target "${DATA_DIR}/${split}.de.jsonl" \
      --tokenizer "${tokenizer}" \
      --max-length "${MAX_LENGTH}" \
      --output "${output}"
  done
}

launch_training() {
  local kind="$1"
  local data="artifacts/translation/wmt14_${kind}_${VOCAB_SIZE}"
  local output="artifacts/translation_runs/wmt14_transformer_base_${kind}_seed${SEED}"
  local command=(
    python -m experiment.train_translation
    --train-data "${data}/train"
    --validation-data "${data}/validation"
    --output "${output}"
    --steps "${STEPS}"
    --tokens-per-batch "${TOKENS_PER_BATCH}"
    --seed "${SEED}"
    --save-steps "${SAVE_STEPS}"
    --resume auto
  )
  if (( NUM_GPUS > 1 )); then
    command=(torchrun --standalone --nproc-per-node "${NUM_GPUS}" -m experiment.train_translation "${command[@]:3}")
  fi
  run_command "${command[@]}"

  run_command python -m experiment.evaluate_translation \
    --run "${output}" \
    --tokenizer "artifacts/tokenizers/wmt14_${kind}_${VOCAB_SIZE}" \
    --source "${DATA_DIR}/test.en.jsonl" \
    --reference "${DATA_DIR}/test.de.jsonl" \
    --output "reports/wmt14_transformer_base_${kind}_seed${SEED}.json"
}

prepare_tokenizer bpe
prepare_tokenizer prefix
prepare_encoded_data bpe
prepare_encoded_data prefix

# These are deliberately separate, sequential jobs so each gets the same hardware budget.
launch_training bpe
launch_training prefix
