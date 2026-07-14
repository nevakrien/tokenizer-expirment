# tokenizer-expirment
trying new tokenizers for NLP since BPE has obvious flaws (for example its not injective).
we seen recently a lot of tricks to try and be more token effishent, for example using mandarin instead of english.
this DOES actually work which implies something is at least somewhat broken about the current encoding.

## setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[test,bpe,clm,data]'
pytest -q
```

## what is currently reproducible

This repo can currently run controlled causal-LM comparisons and tokenizer-only comparisons:

| Target | Status | What to compare |
| --- | --- | --- |
| Small controlled LM | Implemented | BPE vs prefix tree at equal vocab, architecture, data, context, seed and steps |
| GPT-1-style tokenizer setup | Tokenizer-only implemented | 40,000-ID BPE vs prefix tree |
| GPT-2-small-style tokenizer setup | Tokenizer-only implemented | 50,257-ID BPE vs prefix tree |
| Full GPT-1/GPT-2 paper-scale training | Command presets only | Requires original-scale data/compute to call it a full reproduction |
| Attention Is All You Need translation | Not implemented yet | `train_translation.py` is still deferred |

Do not compare token perplexity across tokenizers as the main result. Use held-out tokenization metrics and byte-normalized LM metrics. `experiment.evaluate_clm` reports `bits_per_byte_approx` from raw validation bytes; it also reports dropped packed tokens so you can judge packing error.

## script presets

Run a tiny local check:

```bash
scripts/run_paper_reproduction.sh smoke
```

Run a controlled 8K-vocab WikiText CLM comparison for BPE vs prefix tree:

```bash
MAX_DOCS=10000 MAX_STEPS=10000 SEEDS='0 1 2' scripts/run_paper_reproduction.sh small-clm-wikitext
```

Use `MAX_DOCS=0` to import the full split instead of the default subset:

```bash
MAX_DOCS=0 MAX_STEPS=100000 SEEDS='0 1 2' scripts/run_paper_reproduction.sh small-clm-wikitext
```

Run GPT-1-style tokenizer-only comparison:

```bash
MAX_DOCS=0 scripts/run_paper_reproduction.sh gpt1-tokenizers
```

Run GPT-2-small-style tokenizer-only comparison:

```bash
MAX_DOCS=0 scripts/run_paper_reproduction.sh gpt2-small-tokenizers
```

Useful script environment variables:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `MAX_DOCS` | `10000` | Documents imported per WikiText split. `0` means full split. |
| `SEEDS` | `0 1 2` | Seeds for CLM training. |
| `MAX_STEPS` | `10000` | CLM optimizer steps. |
| `BATCH_SIZE` | `8` | Per-device train batch size. |
| `GRAD_ACCUM` | `1` | Gradient accumulation steps. |
| `VERIFY_RANDOM_CASES` | `100000` | Prefix tokenizer random round-trip checks. |

## manual controlled CLM commands

Import WikiText-103 train and validation splits:

```bash
PYTHONPATH=src python -m experiment.import_hf_dataset --dataset Salesforce/wikitext --name wikitext-103-raw-v1 --split train --output data/imported/wikitext103/train.jsonl --max-docs 10000 --min-chars 20
PYTHONPATH=src python -m experiment.import_hf_dataset --dataset Salesforce/wikitext --name wikitext-103-raw-v1 --split validation --output data/imported/wikitext103/validation.jsonl --max-docs 10000 --min-chars 20
```

Train matched 8K tokenizers:

```bash
PYTHONPATH=src python -m experiment.train_tokenizer --type bpe --dataset-config configs/data/wikitext103_train_subset.json --vocab-size 8192 --output artifacts/tokenizers/wikitext103_bpe_8192
PYTHONPATH=src python -m experiment.train_tokenizer --type prefix_tree --algorithm corpus_count_batched --dataset-config configs/data/wikitext103_train_subset.json --vocab-size 8192 --output artifacts/tokenizers/wikitext103_prefix_8192
PYTHONPATH=src python -m experiment.verify_tokenizer artifacts/tokenizers/wikitext103_prefix_8192 --random-cases 100000
```

Analyze tokenization on held-out validation data:

```bash
PYTHONPATH=src python -m experiment.analyze_tokenizers --tokenizers artifacts/tokenizers/wikitext103_bpe_8192 artifacts/tokenizers/wikitext103_prefix_8192 --dataset-config configs/data/wikitext103_validation_subset.json --output reports/wikitext103_tokenizers_8192.json
```

Preprocess both tokenizers with the same context length:

```bash
PYTHONPATH=src python -m experiment.preprocess --tokenizer artifacts/tokenizers/wikitext103_bpe_8192 --dataset-config configs/data/wikitext103_train_subset.json --context-length 512 --output artifacts/blocks/wikitext103_bpe_8192_ctx512
PYTHONPATH=src python -m experiment.preprocess --tokenizer artifacts/tokenizers/wikitext103_prefix_8192 --dataset-config configs/data/wikitext103_train_subset.json --context-length 512 --output artifacts/blocks/wikitext103_prefix_8192_ctx512
```

Train equal small GPT-style models from scratch:

```bash
PYTHONPATH=src python -m experiment.train_clm --train-data artifacts/blocks/wikitext103_bpe_8192_ctx512 --output artifacts/clm/wikitext103_bpe_8192_seed0 --vocab-size 8192 --context-length 512 --layers 6 --hidden-size 384 --heads 6 --ffn-size 1536 --seed 0 --max-steps 10000 --logging-steps 100 --eval-steps 500 --save-steps 1000
PYTHONPATH=src python -m experiment.train_clm --train-data artifacts/blocks/wikitext103_prefix_8192_ctx512 --output artifacts/clm/wikitext103_prefix_8192_seed0 --vocab-size 8192 --context-length 512 --layers 6 --hidden-size 384 --heads 6 --ffn-size 1536 --seed 0 --max-steps 10000 --logging-steps 100 --eval-steps 500 --save-steps 1000
```

Evaluate both models on the same raw validation documents:

```bash
PYTHONPATH=src python -m experiment.evaluate_clm --model artifacts/clm/wikitext103_bpe_8192_seed0 --tokenizer artifacts/tokenizers/wikitext103_bpe_8192 --dataset-config configs/data/wikitext103_validation_subset.json --context-length 512 --output reports/wikitext103_bpe_8192_seed0_eval.json
PYTHONPATH=src python -m experiment.evaluate_clm --model artifacts/clm/wikitext103_prefix_8192_seed0 --tokenizer artifacts/tokenizers/wikitext103_prefix_8192 --dataset-config configs/data/wikitext103_validation_subset.json --context-length 512 --output reports/wikitext103_prefix_8192_seed0_eval.json
```

## paper-style tokenizer presets

GPT-1-style vocabulary size:

```bash
PYTHONPATH=src python -m experiment.train_tokenizer --type bpe --dataset-config configs/data/wikitext103_train_subset.json --vocab-size 40000 --output artifacts/tokenizers/wikitext103_bpe_40000
PYTHONPATH=src python -m experiment.train_tokenizer --type prefix_tree --algorithm corpus_count_batched --dataset-config configs/data/wikitext103_train_subset.json --vocab-size 40000 --output artifacts/tokenizers/wikitext103_prefix_40000
PYTHONPATH=src python -m experiment.analyze_tokenizers --tokenizers artifacts/tokenizers/wikitext103_bpe_40000 artifacts/tokenizers/wikitext103_prefix_40000 --dataset-config configs/data/wikitext103_validation_subset.json --output reports/wikitext103_tokenizers_40000.json
```

GPT-2-small vocabulary size:

```bash
PYTHONPATH=src python -m experiment.train_tokenizer --type bpe --dataset-config configs/data/wikitext103_train_subset.json --vocab-size 50257 --output artifacts/tokenizers/wikitext103_bpe_50257
PYTHONPATH=src python -m experiment.train_tokenizer --type prefix_tree --algorithm corpus_count_batched --dataset-config configs/data/wikitext103_train_subset.json --vocab-size 50257 --output artifacts/tokenizers/wikitext103_prefix_50257
PYTHONPATH=src python -m experiment.analyze_tokenizers --tokenizers artifacts/tokenizers/wikitext103_bpe_50257 artifacts/tokenizers/wikitext103_prefix_50257 --dataset-config configs/data/wikitext103_validation_subset.json --output reports/wikitext103_tokenizers_50257.json
```

## smoke commands

```bash
PYTHONPATH=src python -m experiment.train_tokenizer --type prefix_tree --dataset-config configs/data/sample.json --vocab-size 1025 --output artifacts/tokenizers/sample_prefix
PYTHONPATH=src python -m experiment.train_tokenizer --type bpe --dataset-config configs/data/sample.json --vocab-size 1025 --output artifacts/tokenizers/sample_bpe
PYTHONPATH=src python -m experiment.analyze_tokenizers --tokenizers artifacts/tokenizers/sample_prefix artifacts/tokenizers/sample_bpe --dataset-config configs/data/sample.json --output reports/sample_tokenization.json
PYTHONPATH=src python -m experiment.preprocess --tokenizer artifacts/tokenizers/sample_prefix --dataset-config configs/data/sample.json --context-length 8 --output artifacts/blocks/sample_prefix
PYTHONPATH=src python -m experiment.train_clm --train-data artifacts/blocks/sample_prefix --output artifacts/clm/sample_prefix --vocab-size 1025 --context-length 8 --layers 2 --hidden-size 128 --heads 4 --ffn-size 512 --max-steps 10
```

## real-data debugging run

```bash
PYTHONPATH=src python -m experiment.import_hf_dataset --dataset Salesforce/wikitext --name wikitext-103-raw-v1 --split train --output data/imported/wikitext103/train.jsonl --max-docs 10000 --min-chars 20
PYTHONPATH=src python -m experiment.train_tokenizer --type prefix_tree --dataset-config configs/data/wikitext103_train_subset.json --vocab-size 8192 --output artifacts/tokenizers/wikitext103_prefix_8192
PYTHONPATH=src python -m experiment.preprocess --tokenizer artifacts/tokenizers/wikitext103_prefix_8192 --dataset-config configs/data/wikitext103_train_subset.json --context-length 128 --output artifacts/blocks/wikitext103_prefix_8192_ctx128
PYTHONPATH=src python -m experiment.train_clm --train-data artifacts/blocks/wikitext103_prefix_8192_ctx128 --output artifacts/clm/wikitext103_prefix_8192_smoke --vocab-size 8192 --context-length 128 --layers 2 --hidden-size 128 --heads 4 --ffn-size 512 --max-steps 20 --logging-steps 5 --save-steps 20
```
