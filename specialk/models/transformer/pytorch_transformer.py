"""
PyTorch native implementation of the Transformer.

Included to sanity check the remaining implementations.
This was based on this implementation:

https://github.com/pytorch/examples/blob/main/word_language_model/model.py
https://pytorch.org/tutorials/beginner/translation_transformer.html
"""

import torch
from jaxtyping import Int, Float, Bool
import numpy as np
from torch import Tensor, LongTensor
import torch.nn as nn
import torch.nn.functional as F
from specialk.models.mt_model import NMTModule
from specialk.models.transformer.Optim import ScheduledOptim
from specialk.models.transformer.pos_encoders import PositionalEncoder
import specialk.core.constants as Constants
from specialk.core.utils import log


class PyTorchTransformerModel(nn.Transformer):
    def __init__(
        self,
        vocab_size: int,
        max_seq_length: int = 100,
        dim_model: int = 512,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout=0.1,
        decoder_generator_weight_sharing=True,
        name: str = "PyTorchTransformer",
        batch_first: bool = True,
        **kwargs,
    ):
        """PyTorch native implementation of a Transformer (see parent class).

        Args:
            vocab_size (int): Vocabulary size of tokenizer.
            max_seq_length (int, optional): Maximum sequence length. Defaults to 100.
            dim_model (int, optional): The number of expected features in the encoder/decoder inputs. Defaults to 512.
            n_heads (int, optional): The number of self-attention heads. Defaults to 8.
            dim_feedforward (int, optional): Dimension of the FFM. Defaults to 2048.
            num_encoder_layers (int, optional): Number of attn layers in the encoder. Defaults to 6.
            num_decoder_layers (int, optional): Number of attn layers in the decoder. Defaults to 6.
            dropout (float, optional): Dropout value. Defaults to 0.1.
            decoder_generator_weight_sharing (bool, optional): If set, shares weight between deocder and generator. Defaults to True.
        """
        super(PyTorchTransformerModel, self).__init__(
            d_model=dim_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.name = name
        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.input_emb = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=dim_model
        )
        self.output_emb = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=dim_model
        )
        self.pos_encoder = PositionalEncoder(
            dim_model=dim_model, max_seq_length=max_seq_length
        )
        self.max_seq_length = max_seq_length
        self.generator = nn.Linear(dim_model, vocab_size)
        self.tgt_mask = None
        self.decoder_generator_weight_sharing = decoder_generator_weight_sharing
        self.x_logit_scale = 1.0
        self.dropout = dropout
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz: int) -> LongTensor:
        """Generate square causal mask for the sequence."""
        return torch.log(
            torch.tril(torch.ones(sz, sz, device=self.generator.weight.device))
        )

    def generate_square_subsequent_mask(self, seq_length: int) -> LongTensor:
        mask = (
            torch.triu(
                torch.ones(
                    (seq_length, seq_length), device=self.generator.weight.device
                )
            )
            == 1
        ).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def create_pad_mask(self, x: Tensor, pad_token: int = Constants.PAD) -> Tensor:
        """Return tensor of the same shape to mask out padding tokens."""
        return x == pad_token

    def init_weights(self):
        """Initiate all weight parameters with Kaiming Uniformity."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p)

    def forward(
        self,
        x: Int[Tensor, "batch_size seq_len"],
        y: Int[Tensor, "batch_size seq_len"],
    ) -> Float[Tensor, "batch_size seq_len generator"]:
        """_summary_

        Args:
            x (LongTensor): Input sequence to perform translation.
            has_mask (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        x_pad_mask, y_pad_mask = self.create_pad_mask(x), self.create_pad_mask(y)
        x_mask = torch.zeros(
            (self.max_seq_length, self.max_seq_length), device=x.device
        ).type(torch.bool)
        y_mask = self.generate_square_subsequent_mask(self.max_seq_length)

        x: Float[Tensor, "batch seq_len d_embed"] = self.pos_encoder(
            self.input_emb(x) * np.sqrt(self.dim_model)
        )
        y: Float[Tensor, "batch seq_len d_embed"] = self.pos_encoder(
            self.output_emb(y) * np.sqrt(self.dim_model)
        )
        y_hat = super().forward(
            src=x,
            tgt=y,
            src_mask=x_mask,
            tgt_mask=y_mask,
            src_key_padding_mask=x_pad_mask,
            tgt_key_padding_mask=y_pad_mask,
            memory_key_padding_mask=x_pad_mask,
        )

        y_hat = self.generator(y_hat)
        return F.log_softmax(y_hat, dim=-1)

    def encode(
        self, x: Float[Tensor, "batch seq_len"], x_mask: Bool[Tensor, "seq_len seq_len"]
    ) -> Tensor:
        """Split encoder and decoder runs."""
        x_emb: Float[Tensor, "batch seq_len d_embed"] = self.input_emb(x) * np.sqrt(
            self.dim_model
        )
        x_emb: Float[Tensor, "batch seq_len d_embed"] = self.pos_encoder(x_emb)
        z: Float[Tensor, "batch seq_len d_model"] = self.encoder(
            self.pos_encoder(self.input_emb(x)), mask=x_mask
        )
        return z

    def decode(
        self,
        y: Float[Tensor, "batch seq_len d_model"],
        memory: Float[Tensor, "batch cur_len d_model"],
        y_mask: Int[Tensor, ""],
    ) -> Tensor:
        """Run decoder stage. This is needed for different decoding strategies."""
        y_emb = self.output_emb(y)
        y_emb = self.pos_encoder(y_emb)
        return self.decoder(y, memory, y_mask)


class PyTorchTransformerModule(NMTModule):
    def __init__(self, n_warmup_steps: int = 4000, **kwargs):
        super().__init__(**kwargs)
        self.n_warmup_steps = n_warmup_steps
        self.model = PyTorchTransformerModel(
            vocab_size=self.vocabulary_size,
            batch_first=True,
            **kwargs,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            filter(lambda x: x.requires_grad, self.model.parameters()),
            betas=(0.9, 0.98),
            eps=1e-09,
            lr=0.0001,
        )
        # TODO implement NoamOptimizer instead.
        return ScheduledOptim(
            optimizer=torch.optim.Adam(
                filter(lambda x: x.requires_grad, self.model.parameters()),
                betas=(0.9, 0.98),
                eps=1e-09,
                lr=0.01,
            ),
            d_model=self.model.dim_model,
            n_warmup_steps=self.n_warmup_steps,
        )
