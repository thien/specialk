"""Load the 2019 OpenNMT-style checkpoints into the modern Transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

import specialk.core.constants as Constants
from specialk.models.transformer.torch.modules import _attend, _project
from specialk.models.transformer.torch.pytorch_transformer import (
    PyTorchTransformerModel,
    PyTorchTransformerModule,
)


@dataclass(frozen=True)
class TokenIds:
    """The numeric special tokens the beam search needs."""

    PAD: int = Constants.PAD
    EOS: int = Constants.EOS


def _strip_module(state_dict: dict) -> dict:
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def _checkpoint(path: Path, weights_only: bool = False) -> dict:
    return torch.load(path, map_location="cpu", weights_only=weights_only)


class LegacyTransformerModel(PyTorchTransformerModel):
    """Modern Transformer weights and beam API for a 2019 checkpoint."""

    @classmethod
    def from_checkpoints(
        cls, encoder_path: Path, decoder_path: Path
    ) -> tuple["LegacyTransformerModel", SimpleNamespace]:
        encoder_checkpoint = _checkpoint(encoder_path)
        decoder_checkpoint = _checkpoint(decoder_path)
        settings = encoder_checkpoint["settings"]
        if settings.proj_share_weight or settings.embs_share_weight:
            raise ValueError("legacy checkpoints with tied embeddings are not supported")
        if settings.d_model // settings.n_head != settings.d_k:
            raise ValueError("only head_dim == d_model / n_head is supported")

        model = cls(
            vocab_size=settings.src_vocab_size,
            decoder_vocab_size=settings.tgt_vocab_size,
            max_seq_length=settings.max_token_seq_len,
            dim_model=settings.d_model,
            n_heads=settings.n_head,
            dim_feedforward=settings.d_inner_hid,
            num_encoder_layers=settings.layers,
            num_decoder_layers=settings.layers,
            dropout=settings.dropout,
            decoder_generator_weight_sharing=False,
            batch_first=True,
        )
        # The legacy model has no final encoder/decoder LayerNorm.
        model.encoder.norm = nn.Identity()
        model.decoder.norm = nn.Identity()

        encoder = _strip_module(encoder_checkpoint["model"])
        decoder = _strip_module(decoder_checkpoint["model"])
        scale = math.sqrt(settings.d_model)
        state = {
            "input_emb.weight": encoder["src_word_emb.weight"] / scale,
            "output_emb.weight": decoder["tgt_word_emb.weight"] / scale,
            "pos_encoder.pos_enc": encoder["position_enc.weight"][
                1 : 1 + settings.max_token_seq_len
            ].unsqueeze(0),
            "generator.weight": decoder_checkpoint["generator"]["weight"],
            "generator.bias": torch.zeros(settings.tgt_vocab_size),
        }
        for i in range(settings.layers):
            source = f"layer_stack.{i}"
            target = f"encoder.layers.{i}"
            state[f"{target}.self_attn.in_proj_weight"] = torch.cat(
                [encoder[f"{source}.slf_attn.{p}.weight"] for p in ("w_qs", "w_ks", "w_vs")],
                dim=0,
            )
            state[f"{target}.self_attn.in_proj_bias"] = torch.cat(
                [encoder[f"{source}.slf_attn.{p}.bias"] for p in ("w_qs", "w_ks", "w_vs")],
                dim=0,
            )
            state[f"{target}.self_attn.out_proj.weight"] = encoder[f"{source}.slf_attn.fc.weight"]
            state[f"{target}.self_attn.out_proj.bias"] = encoder[f"{source}.slf_attn.fc.bias"]
            state[f"{target}.linear1.weight"] = encoder[f"{source}.pos_ffn.w_1.weight"].squeeze(-1)
            state[f"{target}.linear1.bias"] = encoder[f"{source}.pos_ffn.w_1.bias"]
            state[f"{target}.linear2.weight"] = encoder[f"{source}.pos_ffn.w_2.weight"].squeeze(-1)
            state[f"{target}.linear2.bias"] = encoder[f"{source}.pos_ffn.w_2.bias"]
            state[f"{target}.norm1.weight"] = encoder[f"{source}.slf_attn.layer_norm.weight"]
            state[f"{target}.norm1.bias"] = encoder[f"{source}.slf_attn.layer_norm.bias"]
            state[f"{target}.norm2.weight"] = encoder[f"{source}.pos_ffn.layer_norm.weight"]
            state[f"{target}.norm2.bias"] = encoder[f"{source}.pos_ffn.layer_norm.bias"]

            target = f"decoder.layers.{i}"
            state[f"{target}.self_attn.in_proj_weight"] = torch.cat(
                [decoder[f"{source}.slf_attn.{p}.weight"] for p in ("w_qs", "w_ks", "w_vs")],
                dim=0,
            )
            state[f"{target}.self_attn.in_proj_bias"] = torch.cat(
                [decoder[f"{source}.slf_attn.{p}.bias"] for p in ("w_qs", "w_ks", "w_vs")],
                dim=0,
            )
            state[f"{target}.self_attn.out_proj.weight"] = decoder[f"{source}.slf_attn.fc.weight"]
            state[f"{target}.self_attn.out_proj.bias"] = decoder[f"{source}.slf_attn.fc.bias"]
            state[f"{target}.multihead_attn.in_proj_weight"] = torch.cat(
                [decoder[f"{source}.enc_attn.{p}.weight"] for p in ("w_qs", "w_ks", "w_vs")],
                dim=0,
            )
            state[f"{target}.multihead_attn.in_proj_bias"] = torch.cat(
                [decoder[f"{source}.enc_attn.{p}.bias"] for p in ("w_qs", "w_ks", "w_vs")],
                dim=0,
            )
            state[f"{target}.multihead_attn.out_proj.weight"] = decoder[f"{source}.enc_attn.fc.weight"]
            state[f"{target}.multihead_attn.out_proj.bias"] = decoder[f"{source}.enc_attn.fc.bias"]
            state[f"{target}.linear1.weight"] = decoder[f"{source}.pos_ffn.w_1.weight"].squeeze(-1)
            state[f"{target}.linear1.bias"] = decoder[f"{source}.pos_ffn.w_1.bias"]
            state[f"{target}.linear2.weight"] = decoder[f"{source}.pos_ffn.w_2.weight"].squeeze(-1)
            state[f"{target}.linear2.bias"] = decoder[f"{source}.pos_ffn.w_2.bias"]
            state[f"{target}.norm1.weight"] = decoder[f"{source}.slf_attn.layer_norm.weight"]
            state[f"{target}.norm1.bias"] = decoder[f"{source}.slf_attn.layer_norm.bias"]
            state[f"{target}.norm2.weight"] = decoder[f"{source}.enc_attn.layer_norm.weight"]
            state[f"{target}.norm2.bias"] = decoder[f"{source}.enc_attn.layer_norm.bias"]
            state[f"{target}.norm3.weight"] = decoder[f"{source}.pos_ffn.layer_norm.weight"]
            state[f"{target}.norm3.bias"] = decoder[f"{source}.pos_ffn.layer_norm.bias"]

        model.load_state_dict(state, strict=True)
        model.register_buffer(
            "dec_pos_enc",
            decoder["position_enc.weight"][1 : 1 + settings.max_token_seq_len].unsqueeze(0),
        )
        model.eval()
        return model, settings

    def encode(self, x: torch.Tensor, x_pad_mask: torch.Tensor | None = None):
        x_pad_mask = self.create_pad_mask(x) if x_pad_mask is None else x_pad_mask
        keep = (~x_pad_mask).unsqueeze(-1)
        z = self.pos_encoder(self.input_emb(x) * math.sqrt(self.dim_model))
        for layer in self.encoder.layers:
            z = layer(z, src_key_padding_mask=x_pad_mask) * keep
        return z, x_pad_mask

    def decode_step(self, y, memory_kv, x_pad_mask=None, kv_cache=None):
        pos = 0 if kv_cache is None else kv_cache.size(-2)
        x = self.output_emb(y) * math.sqrt(self.dim_model) + self.dec_pos_enc[:, pos : pos + 1]
        new_cache = []
        for n, layer in enumerate(self.decoder.layers):
            self_attn, cross_attn = layer.self_attn, layer.multihead_attn
            k, v = _project(self_attn, x, 1), _project(self_attn, x, 2)
            if kv_cache is not None:
                k = torch.cat([kv_cache[:, n, 0], k], dim=-2)
                v = torch.cat([kv_cache[:, n, 1], v], dim=-2)
            new_cache.append(torch.stack([k, v], dim=1))
            x = layer.norm1(
                x + layer.dropout1(_attend(self_attn, _project(self_attn, x, 0), k, v))
            )
            x = layer.norm2(
                x
                + layer.dropout2(
                    _attend(
                        cross_attn,
                        _project(cross_attn, x, 0),
                        memory_kv[:, n, 0],
                        memory_kv[:, n, 1],
                        x_pad_mask,
                    )
                )
            )
            x = layer.norm3(x + layer._ff_block(x))
        return x, torch.stack(new_cache, dim=1)

    def forward_logits(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        memory, memory_pad = self.encode(x)
        y = y[:, :-1]
        target = self.output_emb(y) * math.sqrt(self.dim_model) + self.dec_pos_enc[:, : y.size(1)]
        target_pad = y.eq(Constants.PAD)
        mask = self.generate_square_subsequent_mask(y.size(1)).bool()
        keep = (~target_pad).unsqueeze(-1)
        for layer in self.decoder.layers:
            target = (
                layer(
                    target,
                    memory,
                    tgt_mask=mask,
                    tgt_key_padding_mask=target_pad,
                    memory_key_padding_mask=memory_pad,
                )
                * keep
            )
        return self.generator(target)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(x, y)
        sos = F.one_hot(y[:, 0], num_classes=self.generator.out_features).unsqueeze(1)
        return torch.cat([sos.to(logits.dtype), F.log_softmax(logits, dim=-1)], dim=1)


def load_legacy_module(
    encoder_path: Path,
    decoder_path: Path,
    tokenizer: TokenIds | None = None,
    decoder_tokenizer: TokenIds | None = None,
    device: str | torch.device = "cpu",
) -> tuple[PyTorchTransformerModule, SimpleNamespace]:
    """Build the modern beam-search module around a 2019 checkpoint pair."""
    model, settings = LegacyTransformerModel.from_checkpoints(encoder_path, decoder_path)
    module = PyTorchTransformerModule(
        vocabulary_size=settings.src_vocab_size,
        sequence_length=settings.max_token_seq_len,
        tokenizer=tokenizer or TokenIds(),
        decoder_vocabulary_size=settings.tgt_vocab_size,
        decoder_tokenizer=decoder_tokenizer or TokenIds(),
        dim_model=settings.d_model,
        n_heads=settings.n_head,
        dim_feedforward=settings.d_inner_hid,
        num_encoder_layers=settings.layers,
        num_decoder_layers=settings.layers,
        dropout=settings.dropout,
        decoder_generator_weight_sharing=False,
    )
    module.model = model.to(device)
    module.eval()
    return module, settings
