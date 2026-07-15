# Compression-Tree Tokenizer Experiment

## Implementation status

This plan has been split into implementation subtasks. The repository now contains a working tokenizer-only v1 focused on correctness, serialization and analysis. Neural training-runner integration is intentionally deferred until tokenizer-only validation produces useful results.

### Completed v1 subtasks

1. Project scaffold
   - Added `pyproject.toml` with `src/` package layout and pytest configuration.
   - Added `src/prefix_tokenizer/` and `src/experiment/` packages.

2. Prefix-tree tokenizer core
   - Implemented complete 256-way trie nodes and vocabulary layout calculation.
   - Implemented deterministic phrase-token ID assignment.
   - Implemented byte and UTF-8 text encode/decode APIs.
   - Implemented bounded byte-tail tokens for documents ending inside an internal trie state.
   - Implemented reserved-ID rejection during decoding.

3. Tree builders
   - Implemented memoryless byte Tunstall-style builder.
   - Implemented batched corpus-count builder using emitted-leaf frequency.
   - Implemented deterministic candidate ordering.

4. Serialization
   - Implemented `save_pretrained` / `from_pretrained`.
   - Exported `tokenizer.json`, `phrases.bin`, `tree.bin` and `metadata.json`.
   - Included corpus fingerprint helper and construction metadata support.

5. Tokenizer-only commands
   - `python -m experiment.train_tokenizer`
   - `python -m experiment.verify_tokenizer`
   - `python -m experiment.analyze_tokenizers`
   - `python -m experiment.preprocess`
   - Added reference byte-BPE training/loading through `--type bpe`.

6. Tests
   - Exhaustive byte round trips for lengths 0, 1 and 2.
   - Random byte round trips.
   - UTF-8 text round trips.
   - Tail-token framing.
   - Tree invariants and reserved-ID rejection.
   - Serialization round trip.
   - Phrase-token sequence uniqueness smoke test.
   - Reference byte-BPE train/save/load byte round trip.

7. Tokenizer analysis and examples
   - Added active-ID utilization, adjacent-pair utilization, pair entropy and unigram frequency metrics.
   - Added checked-in sample corpus and dataset config for reproducible smoke commands.
   - Added multilingual/grouped reporting to tokenizer analysis configs.

### Verification run

1. Unit tests: `pytest -q`
   - Result: `9 passed`.

2. CLI smoke test with `PYTHONPATH=src`
   - Trained a 1025-ID prefix tokenizer on a tiny corpus.
   - Verified exhaustive short-byte and random round trips.
   - Wrote tokenization analysis JSON.
   - Wrote fixed-length preprocessed JSONL blocks.

### Deferred subtasks

1. Hugging Face `PreTrainedTokenizerBase` compatibility beyond the current lightweight adapter.
2. Reference BPE tokenizer training and comparison.
3. Raw-byte tokenizer baseline command.
4. Full pair-transition utilization and entropy analysis.
5. Spec-matching model training runner. Do not add a placeholder runner that can be mistaken for paper reproduction.
6. Byte-normalized model evaluation runner tied to the spec-matching trainer.
7. Translation runner integration.
8. Large experiment matrix, multiple seeds and paper-scale reproductions.

### Immediate next subtasks

1. Define the exact model-training spec before adding any model runner.
2. Add raw-byte tokenizer baseline command.
3. Expand reference BPE validation against GPT-2 assets where available.

## 1. Purpose

Implement a static, byte-level, variable-to-fixed tokenizer based on a complete prefix parsing tree, then substitute it for BPE in existing Transformer and GPT training runners.

The experiment must compare:

1. The paper’s original or closest available BPE tokenizer.
2. A compression-trained prefix-tree tokenizer.
3. Optionally, a raw-byte tokenizer as a sanity baseline.

The BPE and prefix-tree models must have:

* exactly the same model vocabulary size;
* exactly the same Transformer architecture;
* exactly the same embedding and output-layer dimensions;
* exactly the same raw training and validation documents;
* the same optimizer, schedules and random seeds;
* matched training budgets.

The primary questions are:

1. Does the prefix-tree tokenizer produce fewer tokens per byte?
2. Does it improve model quality at a fixed token-context length?
3. Does it improve quality at fixed compute or fixed raw training bytes?
4. Does its unrestricted, uniquely decodable token composition make next-token modelling easier or harder?
5. Does it reduce tokenization differences between languages?

---

# 2. Exact tokenizer definition

## 2.1 Vocabulary

The tokenizer vocabulary consists of:

* phrase tokens represented by leaves of a complete byte-prefix tree;
* ordinary model-level special tokens such as EOS, BOS and PAD;
* reserved IDs needed to match the exact target vocabulary size.

Every phrase leaf stores:

```python
@dataclass
class PhraseLeaf:
    token_id: int
    phrase: bytes
    corpus_count: int
```

Every internal node has exactly 256 children, one for each possible byte:

```python
class TrieNode:
    children: list["TrieNode"] | None  # exactly 256 when internal
    token_id: int | None              # present only when leaf
    corpus_count: int
```

A node is either:

* a leaf with a token ID and no children; or
* an internal node with all 256 children and no token ID.

This invariant makes the tokenizer total and deterministic for an arbitrarily long byte stream.

## 2.2 Encoding

Encoding traverses input bytes until a leaf is reached:

```python
def encode_bytes(data: bytes, tree: Trie) -> list[int]:
    output = []
    position = 0

    while position < len(data):
        node = tree.root

        while node.children is not None:
            if position >= len(data):
                return flush_final_prefix(node, output)

            node = node.children[data[position]]
            position += 1

        output.append(node.token_id)

    return output
```

Conceptually:

```text
token_id, consumed_bytes = search_tree(input)
output.append(token_id)
input.advance(consumed_bytes)
```

Each token ID represents the byte string on the path from the root to its leaf.

## 2.3 Decoding

Decoding is table lookup and concatenation:

```python
def decode_tokens(token_ids: list[int], phrases: list[bytes]) -> bytes:
    output = bytearray()

    for token_id in token_ids:
        if is_special(token_id):
            handle_special(token_id, output)
        elif is_reserved(token_id):
            raise InvalidTokenError(token_id)
        else:
            output.extend(phrases[token_id])

    return bytes(output)
```

No tree traversal is required during decoding.

## 2.4 Unique-decoding property

For phrase tokens (t_1,\ldots,t_n), decoding is:

[
D(t_1,\ldots,t_n)
=================

P(t_1)\Vert P(t_2)\Vert\cdots\Vert P(t_n).
]

Because the phrase set is the set of leaves of a prefix tree, it is prefix-free. Therefore:

[
D(A)=D(B)\implies A=B
]

for any two phrase-token sequences (A) and (B).

Unlike BPE, there cannot be two token sequences such as:

```text
["a", "bc"]
["ab", "c"]
```

that decode to the same bytes.

Any sequence of valid phrase-token IDs decodes to one unique byte sequence.

## 2.5 End-of-document handling

A document can end while traversal is at an internal node. This must not be handled through arbitrary zero-padding because that would change the decoded text.

Use one of these implementations.

### Recommended implementation: terminal-prefix tokens

After constructing the main tree, identify every internal state at which training documents can end. Allocate terminal-prefix tokens representing the byte phrase from the root to that internal node.

A terminal-prefix token:

* is legal only as the final phrase before EOS;
* decodes to its exact partial path;
* is never used in the middle of a document;
* consumes a vocabulary ID.

Encoding becomes:

```python
if end_of_document and node is internal:
    output.append(node.terminal_token_id)
output.append(EOS_ID)
```

The decoder expands the terminal-prefix token normally and then processes EOS.

If allocating a terminal token for every internal node is too expensive, use the fallback below.

### Simpler fallback: bounded byte-tail tokens

Reserve 256 single-byte tail IDs that may appear only immediately before EOS.

When fewer bytes remain than are needed to reach a normal leaf:

1. stop normal tree traversal;
2. emit those remaining bytes using tail-byte IDs;
3. emit EOS.

This guarantees lossless document framing and adds at most `maximum_tree_depth - 1` tail tokens per document.

These IDs count toward the fixed model vocabulary.

Do not silently pad documents or store an out-of-band decoded length for the main experiment.

---

# 3. Matching an arbitrary vocabulary size

A complete 256-way tree begins with 256 leaves. Expanding one leaf replaces it with 256 children, increasing the number of leaves by 255:

[
L=256+255k.
]

Therefore, not every target vocabulary size can be filled entirely with phrase leaves.

For target model vocabulary size (V):

```python
special_count = number of ordinary model special IDs
tail_count = 256 if using bounded byte-tail tokens else 0

available = V - special_count - tail_count
expansions = floor((available - 256) / 255)
phrase_leaf_count = 256 + 255 * expansions
reserved_count = available - phrase_leaf_count
```

The model must still be configured with exactly `V` embeddings and output logits.

Reserved IDs:

* have ordinary embedding/output rows;
* never appear in training data;
* are masked to negative infinity during generation;
* are rejected by the decoder;
* are included so parameter count is exactly equal between tokenizers.

Report both:

* total model vocabulary size;
* number of active phrase tokens;
* number of tail, special and reserved tokens.

Also implement an optional binary-tree version in which raw input is viewed as bits. Expanding a binary leaf adds one leaf, allowing any exact vocabulary size. This version is secondary because it complicates document framing and permits phrase boundaries inside bytes.

---

# 4. Training the phrase tree

Implement at least two construction algorithms.

## 4.1 Baseline: memoryless byte Tunstall tree

Count global byte frequencies:

[
P(b)=\frac{\operatorname{count}(b)+\alpha}
{\sum_x\operatorname{count}(x)+256\alpha}.
]

Initialize the tree with one leaf per byte.

For each leaf representing phrase (s=b_1\cdots b_m), estimate:

[
P(s)=\prod_i P(b_i).
]

Repeatedly expand the highest-probability leaf until the phrase-leaf budget is reached.

Use a max heap:

```python
heap = initial_256_byte_leaves()

while leaf_count + 255 <= phrase_leaf_budget:
    leaf = heap.pop_max()
    make_internal(leaf)

    for byte in range(256):
        child = make_child(leaf, byte)
        child.score = leaf.score + log_probability[byte]
        heap.push(child)

    leaf_count += 255
```

Store log probabilities to avoid underflow.

This implementation is primarily a correctness baseline. It assumes independent bytes and is not expected to be the best language tokenizer.

## 4.2 Main algorithm: corpus-count tree

Build the tree using counts measured directly from the tokenizer-training corpus.

Start with 256 one-byte leaves.

For each iteration:

1. Tokenize or scan the corpus using the current tree.
2. Count:

   * how often each current leaf is reached;
   * the distribution of the byte following that leaf.
3. Assign each leaf an expansion utility.
4. Expand the leaf with the greatest utility.
5. Repeat until the target leaf count is reached.

The simplest utility is:

[
U(s)=\operatorname{count}(s).
]

A better utility estimates the reduction in expected model tokens:

[
U(s)
====

## \operatorname{count}(s)

\sum_b\operatorname{count}(sb)\cdot C(sb),
]

where (C(sb)) estimates the number of output tokens after expansion.

An acceptable first implementation is simply:

```python
utility[leaf] = number_of_times_leaf_is_emitted
```

Expanding the most frequently emitted leaf converts common continuations into longer phrases.

### Efficient implementation

Do not rescan and retokenize the entire corpus after every single expansion.

Use batched expansion:

1. Scan the corpus with the current tree.
2. collect counts for all leaves and their next bytes;
3. select the top `B` non-overlapping expansion candidates;
4. expand them;
5. rescan.

Start with:

```yaml
expansion_batch_size: 64
```

Benchmark batch sizes of 1, 16, 64 and 256.

The implementation may later use a suffix array, suffix automaton or compressed corpus trie, but these are not required for version 1.

## 4.3 Markov score variant

Implement an optional first-order byte model:

[
P(b_{i+1}\mid b_i).
]

A phrase child is scored using:

[
\log P(sb)
==========

\log P(s)+\log P(b\mid \operatorname{last}(s)).
]

This gives a cheap intermediate baseline between independent-byte Tunstall and direct corpus counting.

## 4.4 Determinism

Tree construction must be reproducible.

Tie-breaking order:

1. greater utility;
2. greater corpus count;
3. shorter phrase;
4. lexicographically smaller phrase bytes;
5. lower node creation index.

Save:

* corpus fingerprint;
* complete configuration;
* vocabulary budget;
* phrase table;
* serialized tree;
* token-ID assignment;
* construction statistics.

---

# 5. Token-ID assignment

Tree shape and token numbering are separate.

After tree construction:

1. collect all phrase leaves;
2. sort them deterministically;
3. assign contiguous token IDs;
4. assign tail IDs;
5. assign paper-required special IDs;
6. assign reserved IDs last.

Recommended phrase-leaf ordering:

```text
descending corpus frequency
then descending phrase length
then lexicographic byte order
```

The numerical ordering should not affect the model materially, but deterministic ordering makes artifacts comparable.

Export:

```text
tokenizer.json
phrases.bin
tree.bin
metadata.json
```

`metadata.json` must contain:

```json
{
  "type": "byte_prefix_tree",
  "model_vocab_size": 50257,
  "phrase_leaf_count": 49981,
  "tail_token_count": 256,
  "special_token_count": 1,
  "reserved_token_count": 19,
  "maximum_phrase_bytes": 0,
  "average_phrase_bytes_training": 0.0,
  "training_corpus_sha256": "...",
  "tree_algorithm": "corpus_count_batched",
  "format_version": 1
}
```

The actual counts must be calculated rather than copied from this example.

---

# 6. Required tokenizer API

Expose a Hugging Face-compatible interface where possible:

```python
class PrefixTreeTokenizer:
    vocab_size: int
    bos_token_id: int | None
    eos_token_id: int
    pad_token_id: int | None
    unk_token_id: None

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False
    ) -> list[int]: ...

    def encode_bytes(
        self,
        data: bytes,
        add_eos: bool = False
    ) -> list[int]: ...

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = False
    ) -> str: ...

    def decode_bytes(
        self,
        token_ids: list[int]
    ) -> bytes: ...

    def save_pretrained(self, path: str) -> None: ...

    @classmethod
    def from_pretrained(cls, path: str): ...
```

Text conversion must be exactly:

```python
data = text.encode("utf-8")
```

Decoding text must use strict UTF-8 by default:

```python
text = data.decode("utf-8", errors="strict")
```

Also support byte decoding because prefixes produced during generation may temporarily be invalid UTF-8.

Do not normalize Unicode unless the reference experiment applies the same normalization to both tokenizers.

---

# 7. Tokenizer correctness tests

All tests must run before any model training.

## 7.1 Exhaustive short-byte tests

For all byte strings of length 0, 1 and 2:

```python
assert decode_bytes(encode_bytes(data)) == data
```

Length 2 requires 65,536 cases and is inexpensive.

Sample length-3 strings randomly.

## 7.2 Random round-trip tests

Generate random byte strings with:

* uniform random bytes;
* ASCII-biased bytes;
* valid random UTF-8;
* corpus samples;
* repeated bytes;
* zero bytes;
* all 256 byte values.

Test at least 100,000 random cases:

```python
assert decode_bytes(encode_bytes(x)) == x
```

## 7.3 Unique token-sequence tests

For a small test tree, exhaustively enumerate valid phrase-token sequences up to length four.

Verify:

```python
if sequence_a != sequence_b:
    assert decode_bytes(sequence_a) != decode_bytes(sequence_b)
```

Exclude special, tail and reserved tokens from this property unless their legal positions are included in the test.

## 7.4 Prefix-tree invariants

Verify:

* every internal node has exactly 256 children;
* every phrase leaf has one token ID;
* no token ID is assigned twice;
* no phrase leaf is an ancestor of another phrase leaf;
* every active ID has a decoder entry;
* reserved IDs never have decoder entries;
* every single-byte input can begin encoding successfully;
* maximum tree depth is below a configured safety limit.

## 7.5 BPE reference tests

For each reference tokenizer:

* encode and decode corpus samples;
* save the reference token counts;
* verify that the runner’s existing tokenizer produces the expected IDs;
* ensure that replacing the tokenizer does not modify dataset ordering or document boundaries.

---

# 8. Tokenization analysis before model training

Build a standalone command:

```bash
python -m prefix_tokenizer.analyze \
    --dataset DATASET \
    --split validation \
    --tokenizers gpt2_bpe,prefix_tree,bytes \
    --output reports/tokenization.json
```

For each tokenizer, measure:

## 8.1 Sequence density

Report:

[
\text{bytes per token}
======================

\frac{\text{original UTF-8 bytes}}
{\text{number of non-special tokens}}.
]

Also report:

* tokens per byte;
* Unicode characters per token;
* words per token;
* tokens per document;
* median and percentile tokenized document lengths;
* average phrase length in bytes;
* phrase-length histogram;
* maximum phrase length.

The most important direct comparison is:

```text
mean_prefix_tokens / mean_bpe_tokens
```

A value below 1 means the prefix tree produces shorter sequences.

## 8.2 Context coverage

For each model context size (C), compute how many source bytes fit in one context:

```text
mean bytes represented by C tokens
median bytes represented by C tokens
p05, p95
```

Use the paper’s original token context length without changing it.

## 8.3 Multilingual analysis

Use the same multilingual evaluation files for all tokenizers.

At minimum include:

* English;
* Chinese;
* Hebrew;
* Arabic;
* Spanish;
* Russian;
* Hindi;
* Japanese;
* source code.

Report bytes per token and Unicode characters per token separately.

Because UTF-8 uses different numbers of bytes for different scripts, do not claim language fairness based solely on characters per token. Show:

* tokens per UTF-8 byte;
* tokens per Unicode scalar;
* tokens per whitespace-delimited word where meaningful;
* compressed/tokenized sequence length relative to BPE.

## 8.4 Pair-transition utilization

For each tokenizer, count observed adjacent token pairs.

Report:

```text
unique observed pairs
unique observed pairs / V²
pair entropy
fraction of token IDs observed
fraction of active IDs observed
```

For BPE, also estimate how many sampled token pairs decode to text that the tokenizer would re-encode into that same pair:

```python
canonical = encode_bytes(decode_bytes([a, b]))
pair_is_canonical = canonical == [a, b]
```

Sample at least one million pairs if exhaustive (V^2) enumeration is too large.

For the prefix-tree phrase tokens:

```python
encode_bytes(decode_bytes([a, b])) == [a, b]
```

must hold for ordinary phrase tokens.

## 8.5 Token-frequency distribution

Report:

* unigram entropy;
* top-token frequency;
* number of tokens covering 50%, 90%, 99% of occurrences;
* least-used active tokens;
* zero-frequency IDs;
* average conditional next-token entropy estimated from bigrams.

This may reveal whether increased token density creates a harder next-token distribution.

---

# 9. Training-runner integration

Do not reimplement GPT or Transformer unless necessary.

Use existing maintained runners and replace only:

1. tokenizer construction/loading;
2. dataset preprocessing;
3. vocabulary-size configuration;
4. generation decoding;
5. evaluation metrics.

## 9.1 Preferred language-model runner

Use Hugging Face Transformers’ causal-language-model training infrastructure for the controlled GPT-style experiments.

The official Transformers examples support causal language-model training and training models from scratch.

Implement a tokenizer adapter compatible with:

```python
PreTrainedTokenizerBase
```

or bypass tokenizer internals by preprocessing the dataset into Arrow fields containing:

```python
{
    "input_ids": list[int],
    "attention_mask": list[int]
}
```

The second option is simpler and reduces integration assumptions.

Instantiate GPT-2 architecture from configuration, not pretrained weights:

```python
config = GPT2Config(
    vocab_size=target_vocab_size,
    n_positions=context_length,
    n_ctx=context_length,
    n_embd=...,
    n_layer=...,
    n_head=...,
    bos_token_id=...,
    eos_token_id=...,
)

model = GPT2LMHeadModel(config)
```

The reference BPE and prefix-tree models must both be initialized from scratch with the same configuration and seed.

Do not reuse pretrained GPT weights, because changing token semantics invalidates the embedding and output layers and makes the comparison uncontrolled.

## 9.2 Optional minimal runner

Use `minGPT` or an equivalent small PyTorch implementation for debugging. The repository exposes a compact GPT implementation and explicitly supports GPT-2’s 50,257 vocabulary and 1,024-token block size.

This is useful for:

* Tiny Shakespeare;
* TinyStories;
* small WikiText experiments;
* debugging generation;
* quick tokenizer comparisons.

Do not use it as the only final experiment unless the reproduction target is explicitly a reduced model.

## 9.3 GPT-2 reference assets

The official OpenAI GPT-2 repository contains the released model code and tokenizer implementation.

Use its tokenizer as a validation reference, but prefer a modern PyTorch runner for new training.

## 9.4 Translation runner

For *Attention Is All You Need*, prefer one of:

1. Tensor2Tensor, for historical fidelity;
2. Fairseq or OpenNMT-py, for easier tokenizer substitution;
3. a maintained PyTorch reproduction with configurable preprocessing.

Tensor2Tensor includes the Transformer implementation and a WMT English-to-German training walkthrough and was the codebase associated with the original implementation.

The integration must preprocess source and target into integer ID files using the selected tokenizer before training.

For a shared source-target vocabulary:

* train one BPE tokenizer on the concatenated source and target tokenizer corpus;
* train one prefix tree on the exact same concatenated bytes;
* use one shared embedding/output vocabulary where the runner supports it.

---

# 10. Dataset preprocessing

## 10.1 Split before tokenizer training

The corpus must be split into:

* tokenizer-training data;
* model-training data;
* validation data;
* test data.

The tokenizer may be trained on the model-training split, but never on validation or test data.

For paper reproduction, follow the paper’s original train/validation/test split when available.

## 10.2 Preserve document boundaries

Encode each document independently:

```text
phrase tokens
optional tail tokens
EOS
```

Do not allow normal phrase tokens to span document boundaries.

For translation, encode each sentence independently.

## 10.3 Packing

After tokenization, pack examples into fixed token-length blocks using the same packing policy for both tokenizers.

Two evaluation regimes are required.

### Fixed token context

Use the paper’s original token context size.

This measures the practical benefit of fitting more raw text into the same number of Transformer positions.

### Fixed raw-byte exposure

Train until each model has consumed the same number of original corpus bytes.

Because the prefix tokenizer may generate fewer tokens, it may require fewer optimizer steps to see the same raw text. Record total FLOPs and wall-clock time.

Also run a fixed-token-budget comparison:

```text
same number of non-padding training tokens
```

This answers a different question: what happens when both models receive the same Transformer compute but different quantities of raw language?

---

# 11. Training experiment matrix

Use at least three seeds for serious comparisons.

## Phase 0: tokenizer-only

Datasets:

* WikiText-103 or TinyStories;
* a fixed multilingual sample;
* WMT/IWSLT source and target text.

Tokenizers:

* paper/reference BPE;
* memoryless Tunstall;
* corpus-count prefix tree;
* raw bytes.

No neural training.

Gate to continue:

* all round trips succeed;
* vocabulary sizes match;
* corpus-count tree produces a meaningful sequence-length reduction or an informative negative result.

## Phase 1: small causal LM

Suggested model:

```yaml
layers: 6
hidden_size: 384
heads: 6
ffn_size: 1536
context_length: 512
dropout: 0.1
vocab_size: 8192 or 16384
```

Use the same exact vocabulary size for all learned tokenizers.

Train:

* three seeds;
* fixed raw-byte budget;
* fixed token budget.

This establishes whether the tokenizer works before expensive runs.

## Phase 2: GPT-2-small architecture

```yaml
layers: 12
hidden_size: 768
heads: 12
context_length: 1024
vocab_size: 50257
```

Compare:

* GPT-2 byte-level BPE;
* corpus-count prefix tree;
* optional memoryless tree.

Use identical configurations and initialization seeds.

## Phase 3: GPT-1-style vocabulary experiment

```yaml
layers: 12
hidden_size: 768
heads: 12
context_length: 512
vocab_size: 40000
```

This isolates whether results depend on the GPT-2 byte-level BPE design.

## Phase 4: Transformer translation

Start with IWSLT English–German for debugging.

Then use WMT 2014 English–German if resources permit.

Compare:

* reference shared BPE;
* shared prefix-tree tokenizer;
* equal vocabulary size;
* identical Transformer-base architecture.

---

# 12. Model evaluation

## 12.1 Do not compare token perplexity directly

Different tokenizers have different token units. Token-level loss and perplexity are not directly comparable.

Calculate total validation negative log-likelihood and normalize it by original bytes:

[
\mathrm{BPB}
============

\frac{-\sum_i \log_2 P(t_i\mid t_{<i})}
{N_{\text{original bytes}}}.
]

Also report:

[
\mathrm{NLL/character}
]

and, where appropriate:

[
\mathrm{NLL/word}.
]

## 12.2 Causal-LM metrics

Report:

* validation bits per byte;
* validation loss per token;
* raw bytes processed;
* tokens processed;
* training FLOPs if available;
* wall-clock training time;
* tokens per second;
* original bytes per second;
* peak memory;
* bytes covered by one context;
* generation validity;
* UTF-8 validity after complete generated documents.

## 12.3 Translation metrics

Report:

* BLEU using one fixed evaluation implementation;
* chrF;
* validation NLL per source/target byte;
* source tokens per sentence;
* target tokens per sentence;
* truncated sentence rate;
* training throughput;
* decoding speed.

## 12.4 Generation validity

The model must never generate reserved IDs because they are masked.

Track:

* terminal/tail-token legality;
* incomplete UTF-8 endings;
* EOS rate;
* malformed special-token ordering;
* proportion of generated byte sequences that decode as valid UTF-8.

It is legal for an intermediate generated byte prefix to be invalid UTF-8. Evaluate validity only at a completed EOS-delimited output.

---

# 13. Ablations

Run these after the main tokenizer works.

## 13.1 Maximum phrase depth

Test:

```text
8 bytes
16 bytes
32 bytes
64 bytes
unbounded with safety cap
```

Long phrases may improve compression but create rare, overly specific tokens.

## 13.2 Tree construction method

Compare:

* independent byte frequencies;
* first-order Markov frequencies;
* direct corpus counts;
* direct corpus counts with batched expansion.

## 13.3 Vocabulary size

Compare equal sizes such as:

```text
8,192
16,384
32,000
40,000
50,257
65,536
```

For every point, compare against a BPE tokenizer trained on the same tokenizer corpus with the same model vocabulary size.

## 13.4 Corpus domain

Train tokenizers on:

* English prose;
* multilingual text;
* source code;
* mixed web text.

Cross-evaluate every tokenizer on every domain.

## 13.5 Explicit decoder-state information

The standard phrase-token representation does not require decoder state: every token ID independently identifies a complete phrase.

Do not add positional decoder-state embeddings in the main experiment.

## 13.6 Pair uniqueness versus compression

Compare the strict prefix-tree vocabulary with a dictionary tokenizer that allows overlapping phrases but chooses a canonical segmentation through dynamic programming.

This tests whether unique token-sequence decoding itself helps, apart from sequence compression.

---

# 14. Commands and configuration

Target command structure:

```bash
# Train reference BPE
python -m experiment.train_tokenizer \
    --type bpe \
    --dataset-config configs/data/fineweb_subset.yaml \
    --vocab-size 50257 \
    --output artifacts/tokenizers/gpt2_bpe

# Train prefix tree
python -m experiment.train_tokenizer \
    --type prefix_tree \
    --algorithm corpus_count_batched \
    --dataset-config configs/data/fineweb_subset.yaml \
    --vocab-size 50257 \
    --tail-byte-tokens \
    --output artifacts/tokenizers/prefix_50257

# Verify tokenizers
python -m experiment.verify_tokenizer \
    artifacts/tokenizers/prefix_50257

# Compare sequence lengths
python -m experiment.analyze_tokenizers \
    --tokenizers \
        artifacts/tokenizers/gpt2_bpe \
        artifacts/tokenizers/prefix_50257 \
    --dataset-config configs/data/evaluation.yaml \
    --output reports/tokenization_50257

# Preprocess model data
python -m experiment.preprocess \
    --tokenizer artifacts/tokenizers/prefix_50257 \
    --dataset-config configs/data/fineweb_subset.yaml \
    --context-length 1024 \
    --output artifacts/datasets/prefix_50257_ctx1024

# Model training and evaluation are intentionally omitted until the runner
# matches the target paper spec closely enough to avoid misleading results.
```

---

# 15. Repository layout

```text
compression-tree-tokenizer/
├── README.md
├── pyproject.toml
├── configs/
│   ├── data/
│   ├── models/
│   ├── tokenizers/
│   └── experiments/
├── src/
│   ├── prefix_tokenizer/
│   │   ├── tree.py
│   │   ├── builder_memoryless.py
│   │   ├── builder_markov.py
│   │   ├── builder_corpus.py
│   │   ├── tokenizer.py
│   │   ├── serialization.py
│   │   ├── framing.py
│   │   └── hf_adapter.py
│   └── experiment/
│       ├── train_tokenizer.py
│       ├── verify_tokenizer.py
│       ├── analyze_tokenizers.py
│       └── preprocess.py
├── tests/
│   ├── test_roundtrip.py
│   ├── test_exhaustive_bytes.py
│   ├── test_unique_sequences.py
│   ├── test_tree_invariants.py
│   ├── test_serialization.py
│   ├── test_framing.py
│   └── test_hf_integration.py
├── artifacts/
├── reports/
└── runs/
```

---

# 16. Required output tables

## Tokenizer-only table

| Tokenizer | Model vocab | Active phrases | Reserved IDs | Bytes/token | Chars/token | Median doc tokens | Pair utilization |
| --------- | ----------: | -------------: | -----------: | ----------: | ----------: | ----------------: | ---------------: |

## Language comparison

| Language | UTF-8 bytes | BPE tokens | Tree tokens | Tree/BPE | BPE chars/token | Tree chars/token |
| -------- | ----------: | ---------: | ----------: | -------: | --------------: | ---------------: |

## Language-model comparison

| Model | Tokenizer | Training bytes | Training tokens | FLOPs | Validation BPB | Bytes/context | Bytes/sec |
| ----- | --------- | -------------: | --------------: | ----: | -------------: | ------------: | --------: |

## Translation comparison

| Tokenizer | Source tokens | Target tokens | BLEU | chrF | Target BPB | Training time |
| --------- | ------------: | ------------: | ---: | ---: | ---------: | ------------: |

---

# 17. Success criteria

The experiment should not define success merely as shorter sequences.

A promising result requires most of the following:

1. Prefix-tree tokenization uses fewer model tokens per raw byte than equal-size BPE.
2. Validation bits per byte are no worse at equal raw-byte training exposure.
3. It improves quality or context coverage at equal Transformer compute.
4. Generation remains stable.
5. It does not create severe rare-token or optimization problems.
6. Multilingual sequence-length disparities decrease.
7. Benefits survive multiple seeds and more than one corpus.

A negative result is also useful if the tokenizer produces shorter sequences but worse bits per byte. That would indicate that linguistic or reusable BPE boundaries aid prediction enough to outweigh their sequence inefficiency.

---

# 18. Canonical paper targets

## A. Attention Is All You Need

Paper:

**Ashish Vaswani et al., “Attention Is All You Need,” 2017.**

Primary task:

* WMT 2014 English–German translation;
* Transformer-base first;
* shared source-target vocabulary;
* target model vocabulary approximately 37,000, matching the experiment described by the paper. The paper reports roughly 4.5 million English–German sentence pairs.

Preferred runner:

* Tensor2Tensor for historical fidelity;
* Fairseq or OpenNMT-py if tokenizer substitution is substantially easier.

Tensor2Tensor contains Transformer training support and a WMT English–German walkthrough.

Comparison:

```text
shared BPE vocabulary: 37,000 model IDs
shared prefix-tree vocabulary: 37,000 model IDs
```

Use IWSLT English–German as a preliminary debugging run.

## B. Improving Language Understanding by Generative Pre-Training

Paper:

**Alec Radford et al., “Improving Language Understanding by Generative Pre-Training,” 2018.**

Target:

```text
model vocabulary size: 40,000
layers: 12
hidden size: 768
attention heads: 12
context length: 512
```

Run the unsupervised causal-language-model phase first. Downstream supervised task reproductions are optional after the tokenizer comparison works.

Use Hugging Face’s OpenAI-GPT architecture implementation or a configuration-equivalent decoder-only Transformer. A hosted OpenAI-GPT model definition remains available through Hugging Face.

Do not initialize from pretrained GPT-1 weights.

## C. Language Models Are Unsupervised Multitask Learners

Paper:

**Alec Radford et al., “Language Models Are Unsupervised Multitask Learners,” 2019.**

Target GPT-2-small architecture:

```text
model vocabulary size: 50,257
layers: 12
hidden size: 768
attention heads: 12
context length: 1,024
```

GPT-2’s released tokenizer vocabulary contains 50,257 IDs, and the official repository contains the model and tokenizer code.

Preferred future runner:

* spec-matching from-scratch causal-LM training;
* optional minGPT debugging runner.

Comparison:

```text
GPT-2 byte-level BPE: 50,257 model IDs
prefix-tree tokenizer: 50,257 model IDs
```

Train both models from random initialization on the same accessible replacement corpus. Do not describe this as a full GPT-2 reproduction unless the original-scale corpus and training budget are actually reproduced.

## Recommended execution order

1. Tokenizer-only analysis at 8K, 40K and 50,257 vocabulary sizes.
2. Small causal LM on TinyStories or WikiText.
3. GPT-2-small architecture on a manageable web-text corpus.
4. IWSLT translation.
5. WMT 2014 Transformer-base.
6. GPT-1-style model.
7. Larger models only after the smaller comparisons show a clear benefit.
