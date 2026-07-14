# tokenizer-expirment
trying new tokenizers for NLP since BPE has obvious flaws (for example its not injective).
we seen recently a lot of tricks to try and be more token effishent, for example using mandarin instead of english.
this DOES actually work which implies something is at least somewhat broken about the current encoding.

## smoke commands

```bash
PYTHONPATH=src python -m experiment.train_tokenizer --type prefix_tree --dataset-config configs/data/sample.json --vocab-size 1025 --output artifacts/tokenizers/sample_prefix
PYTHONPATH=src python -m experiment.train_tokenizer --type bpe --dataset-config configs/data/sample.json --vocab-size 1025 --output artifacts/tokenizers/sample_bpe
PYTHONPATH=src python -m experiment.analyze_tokenizers --tokenizers artifacts/tokenizers/sample_prefix artifacts/tokenizers/sample_bpe --dataset-config configs/data/sample.json --output reports/sample_tokenization.json
PYTHONPATH=src python -m experiment.preprocess --tokenizer artifacts/tokenizers/sample_prefix --dataset-config configs/data/sample.json --context-length 64 --output artifacts/blocks/sample_prefix
PYTHONPATH=src python -m experiment.train_clm --train-data artifacts/blocks/sample_prefix --output artifacts/clm/sample_prefix --vocab-size 1025 --context-length 64 --layers 2 --hidden-size 128 --heads 4 --ffn-size 512 --max-steps 10
```
