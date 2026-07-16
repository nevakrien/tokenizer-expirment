from __future__ import annotations

import json
from pathlib import Path


EOS_TOKEN = "<|endoftext|>"
BOS_TOKEN = "<|startoftext|>"
PAD_TOKEN = "<|pad|>"


def bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for byte in range(256):
        if byte not in bs:
            bs.append(byte)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(item) for item in cs), strict=True))


BYTE_TO_UNICODE = bytes_to_unicode()
UNICODE_TO_BYTE = {value: key for key, value in BYTE_TO_UNICODE.items()}


class ReferenceBPETokenizer:
    def __init__(
        self,
        tokenizer,
        vocab_size: int,
        eos_token_id: int,
        bos_token_id: int | None = None,
        pad_token_id: int | None = None,
        reserved_ids: set[int] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.unk_token_id = None
        self.reserved_ids = reserved_ids or set()
        vocab = tokenizer.get_vocab()
        self._token_by_id = {token_id: token for token, token_id in vocab.items()}

    @classmethod
    def train(
        cls,
        documents: list[bytes],
        vocab_size: int,
        *,
        special_token_count: int = 1,
        show_progress: bool = False,
    ) -> "ReferenceBPETokenizer":
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
        except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
            raise RuntimeError("training BPE tokenizers requires the optional 'tokenizers' package") from exc

        tokenizer = Tokenizer(BPE())
        if special_token_count not in {1, 3}:
            raise ValueError("BPE supports either EOS only or EOS/BOS/PAD special tokens")
        special_tokens = [EOS_TOKEN]
        if special_token_count == 3:
            special_tokens.extend([BOS_TOKEN, PAD_TOKEN])
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            initial_alphabet=list(BYTE_TO_UNICODE.values()),
            max_token_length=32,
            show_progress=show_progress,
        )
        tokenizer.train_from_iterator(
            (_bytes_to_text(document) for document in documents),
            trainer=trainer,
            length=len(documents),
        )
        eos_token_id = tokenizer.token_to_id(EOS_TOKEN)
        if eos_token_id is None:
            raise RuntimeError("BPE trainer did not assign an EOS token")
        actual_vocab_size = tokenizer.get_vocab_size()
        if actual_vocab_size > vocab_size:
            raise ValueError(f"vocab_size must be at least {actual_vocab_size} for byte-BPE")
        reserved_ids = set(range(actual_vocab_size, vocab_size))
        return cls(
            tokenizer,
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            bos_token_id=tokenizer.token_to_id(BOS_TOKEN),
            pad_token_id=tokenizer.token_to_id(PAD_TOKEN),
            reserved_ids=reserved_ids,
        )

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "ReferenceBPETokenizer":
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
            raise RuntimeError("loading BPE tokenizers requires the optional 'tokenizers' package") from exc

        path = Path(path)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        eos_token_id = tokenizer.token_to_id(EOS_TOKEN)
        if eos_token_id is None:
            raise RuntimeError("BPE tokenizer is missing EOS token")
        return cls(
            tokenizer,
            vocab_size=metadata["model_vocab_size"],
            eos_token_id=eos_token_id,
            bos_token_id=tokenizer.token_to_id(BOS_TOKEN),
            pad_token_id=tokenizer.token_to_id(PAD_TOKEN),
            reserved_ids=set(metadata.get("reserved_ids", [])),
        )

    def save_pretrained(self, path: str | Path, *, metadata: dict | None = None) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path / "tokenizer.json"))
        base_metadata = {
            "type": "byte_bpe",
            "format_version": 1,
            "model_vocab_size": self.vocab_size,
            "active_token_count": self.tokenizer.get_vocab_size(),
            "special_token_count": 1 + int(self.bos_token_id is not None) + int(self.pad_token_id is not None),
            "reserved_token_count": len(self.reserved_ids),
            "reserved_ids": sorted(self.reserved_ids),
        }
        if metadata:
            base_metadata.update(metadata)
        (path / "metadata.json").write_text(json.dumps(base_metadata, indent=2, sort_keys=True), encoding="utf-8")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.encode_bytes(text.encode("utf-8"), add_eos=add_special_tokens)

    def encode_bytes(self, data: bytes, add_eos: bool = False) -> list[int]:
        ids = self.tokenizer.encode(_bytes_to_text(data), add_special_tokens=False).ids
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return self.decode_bytes(token_ids, skip_special_tokens=skip_special_tokens).decode("utf-8", errors="strict")

    def decode_bytes(self, token_ids: list[int], skip_special_tokens: bool = False) -> bytes:
        text = []
        for token_id in token_ids:
            if token_id in {self.eos_token_id, self.bos_token_id, self.pad_token_id}:
                if skip_special_tokens:
                    continue
                continue
            if token_id in self.reserved_ids:
                raise ValueError(f"reserved token id cannot be decoded: {token_id}")
            token = self._token_by_id.get(token_id)
            if token is None:
                raise ValueError(f"unknown token id: {token_id}")
            text.append(token)
        return _text_to_bytes("".join(text))


def load_any_tokenizer(path: str | Path):
    path = Path(path)
    metadata_path = path / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("type") == "byte_bpe":
            return ReferenceBPETokenizer.from_pretrained(path)

    from prefix_tokenizer import PrefixTreeTokenizer

    return PrefixTreeTokenizer.from_pretrained(path)


def _bytes_to_text(data: bytes) -> str:
    return "".join(BYTE_TO_UNICODE[byte] for byte in data)


def _text_to_bytes(text: str) -> bytes:
    return bytes(UNICODE_TO_BYTE[char] for char in text)
