from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 37000
    d_model: int = 512
    d_ff: int = 2048
    heads: int = 8
    encoder_layers: int = 6
    decoder_layers: int = 6
    dropout: float = 0.1
    max_length: int = 512
    pad_token_id: int = -1

    def to_dict(self) -> dict:
        return asdict(self)


def build_model(config: TransformerConfig):
    import torch
    from torch import nn

    class SinusoidalEncoding(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            positions = torch.arange(config.max_length, dtype=torch.float32).unsqueeze(1)
            frequencies = torch.exp(torch.arange(0, config.d_model, 2) * (-math.log(10000.0) / config.d_model))
            values = torch.zeros(config.max_length, config.d_model)
            values[:, 0::2] = torch.sin(positions * frequencies)
            values[:, 1::2] = torch.cos(positions * frequencies)
            self.register_buffer("values", values, persistent=False)

        def forward(self, inputs):
            return inputs + self.values[: inputs.shape[1]].to(dtype=inputs.dtype)

    class PaperTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
            self.position = SinusoidalEncoding()
            self.dropout = nn.Dropout(config.dropout)
            self.transformer = nn.Transformer(
                d_model=config.d_model,
                nhead=config.heads,
                num_encoder_layers=config.encoder_layers,
                num_decoder_layers=config.decoder_layers,
                dim_feedforward=config.d_ff,
                dropout=config.dropout,
                activation="relu",
                batch_first=True,
                norm_first=False,
            )
            self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.output.weight = self.embedding.weight

        def _embed(self, token_ids):
            return self.dropout(self.position(self.embedding(token_ids) * math.sqrt(config.d_model)))

        def forward(self, source, target):
            source_padding = source.eq(config.pad_token_id)
            target_padding = target.eq(config.pad_token_id)
            causal_mask = torch.triu(
                torch.ones(target.shape[1], target.shape[1], dtype=torch.bool, device=target.device),
                diagonal=1,
            )
            hidden = self.transformer(
                self._embed(source),
                self._embed(target),
                tgt_mask=causal_mask,
                src_key_padding_mask=source_padding,
                tgt_key_padding_mask=target_padding,
                memory_key_padding_mask=source_padding,
            )
            return self.output(hidden)

    return PaperTransformer()


def paper_learning_rate(step: int, d_model: int = 512, warmup_steps: int = 4000) -> float:
    if step < 1:
        raise ValueError("step must be at least one")
    return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)
