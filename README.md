# tokenizer-expirment
trying new tokenizers for NLP since BPE has obvious flaws (for example its not injective).
we seen recently a lot of tricks to try and be more token effishent, for example using mandarin instead of english.
this DOES actually work which implies something is at least somewhat broken about the current encoding.

## setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[test,bpe,data,translation]'
pytest -q
```

## what is currently reproducible

The repository includes a modern PyTorch reproduction of the Transformer-base WMT 2014 English-German experiment from *Attention Is All You Need*. It runs a matched shared byte-BPE baseline and shared prefix-tree tokenizer experiment.

| Target | Status | What to compare |
| --- | --- | --- |
| GPT-1-style tokenizer setup | Tokenizer-only implemented | 40,000-ID BPE vs prefix tree |
| GPT-2-small-style tokenizer setup | Tokenizer-only implemented | 50,257-ID BPE vs prefix tree |
| Full GPT-1/GPT-2 paper-scale training | Not implemented | Needs a spec-matching trainer before any reproduction claim |
| Attention Is All You Need translation | Implemented | WMT14 En-De, Transformer-base, BPE vs prefix tree |

The WMT runner matches the paper's dataset, Transformer-base dimensions, tied embeddings, sinusoidal positions, optimizer, schedule, dropout, label smoothing, 25K source/target token batches, 100K steps, beam size 4, length penalty 0.6, and five-checkpoint averaging. It uses maintained PyTorch code and this project's reversible byte-BPE rather than the original Tensor2Tensor/Sennrich preprocessing code. That tokenizer difference is intentional and recorded in run metadata.

## WMT14 paper reproduction

Install dependencies, then run both complete training jobs:

```bash
scripts/run_wmt14_reproduction.sh
```

The script downloads `wmt/wmt14` English-German, trains both shared 37K tokenizers on the training split only, preprocesses the same aligned pairs, runs the BPE and prefix-tree Transformer-base jobs sequentially, and evaluates both on `newstest2014` with SacreBLEU. Validation uses `newstest2013`.

The script has fixed reproduction settings: the complete dataset, a shared 37K vocabulary, Transformer-base, 25K source and target tokens per global batch, and 100K optimization steps. It uses every visible GPU and runs in full precision. The paper used eight P100 GPUs, so running this requires substantial compute. Interrupted runs resume from their latest checkpoint automatically. Outputs are written under `artifacts/translation_runs/`; BLEU reports are written under `reports/`.

## tokenizer commands

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
```

## real-data debugging run

```bash
PYTHONPATH=src python -m experiment.import_hf_dataset --dataset Salesforce/wikitext --name wikitext-103-raw-v1 --split train --output data/imported/wikitext103/train.jsonl --max-docs 10000 --min-chars 20
PYTHONPATH=src python -m experiment.train_tokenizer --type prefix_tree --dataset-config configs/data/wikitext103_train_subset.json --vocab-size 8192 --output artifacts/tokenizers/wikitext103_prefix_8192
PYTHONPATH=src python -m experiment.preprocess --tokenizer artifacts/tokenizers/wikitext103_prefix_8192 --dataset-config configs/data/wikitext103_train_subset.json --context-length 128 --output artifacts/blocks/wikitext103_prefix_8192_ctx128
```
