"""
PyTorch native implementation of the Transformer.

This is intentional to take advantage of native C level
implementations (as reasonably as possible).
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from schedulefree import AdamWScheduleFree
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LRScheduler

import specialk.core.constants as Constants
from specialk.models.mt_model import NMTModule
from specialk.models.transformer.pos_encoders import PositionalEncoder


class PyTorchTransformerModel(nn.Transformer):
    def __init__(
        self,
        vocab_size: int,
        decoder_vocab_size: Optional[int] = None,
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
            max_seq_length (int, optional): Maximum sequence length.
                Defaults to 100.
            dim_model (int, optional): The number of expected features in the
                encoder/decoder inputs. Defaults to 512.
            n_heads (int, optional): The number of self-attention heads.
                Defaults to 8.
            dim_feedforward (int, optional): Dimension of the FFM. Defaults to 2048.
            num_encoder_layers (int, optional): Number of attn layers in the encoder.
                Defaults to 6.
            num_decoder_layers (int, optional): Number of attn layers in the decoder.
                Defaults to 6.
            dropout (float, optional): Dropout value. Defaults to 0.1.
            decoder_generator_weight_sharing (bool, optional): If set, shares weight
                between deocder and generator. Defaults to True.
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
        self.max_seq_length = max_seq_length
        self.dim_model = dim_model
        self.tgt_mask = None
        self.decoder_generator_weight_sharing = decoder_generator_weight_sharing
        self.x_logit_scale = 1.0
        self.dropout = dropout
        if not decoder_vocab_size:
            decoder_vocab_size = vocab_size

        self.input_emb = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=dim_model
        )
        self.output_emb = nn.Embedding(
            num_embeddings=decoder_vocab_size, embedding_dim=dim_model
        )
        self.pos_encoder = PositionalEncoder(
            dim_model=dim_model, max_seq_length=max_seq_length
        )
        self.generator = nn.Linear(dim_model, decoder_vocab_size)
        self.init_weights()

    def generate_square_subsequent_mask(self, size: int) -> Tensor:
        """Generate square causal mask.

        Top half of the diagonal is -inf, else 0's.

        Parameters:
                length (int): Number of tokens in each sequence in the target batch.
        Returns:
                mask (arr): tgt_mask, looks like [[0., -inf, -inf],
                                                  [0.,   0., -inf],
                                                  [0.,   0.,   0.]]
                            for a size=3.
        """
        return torch.log(
            torch.tril(torch.ones(size, size, device=self.generator.weight.device))
        )

    def create_pad_mask(self, x: Tensor, pad_token: int = Constants.PAD) -> Tensor:
        """Return tensor of the same shape to mask out padding tokens."""
        return x == pad_token

    def init_weights(self):
        """Initiate all weight parameters with Kaiming Uniformity."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p)

    def _forward(
        self,
        x: Int[Tensor, "batch seq"],
        y: Int[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch seq model"]:
        """Runs forward training pass for this seq2seq transformer training.

        Parameters:
            x (Tensor): Input sequence to train.
            y (Tensor): Output sequence to train.

        Returns:
            Tensor: output tokens by model space.
        """

        # make it causal.
        y = y[:, :-1]

        # create masks
        length = self.max_seq_length
        x_pad_mask, y_pad_mask = self.create_pad_mask(x), self.create_pad_mask(y)
        x_mask = torch.zeros((length, length), device=x.device).type(torch.bool)
        y_mask = self.generate_square_subsequent_mask(y.shape[-1]).bool()

        x_emb: Float[Tensor, "batch seq_len d_embed"] = self.pos_encoder(
            self.input_emb(x) * np.sqrt(self.dim_model)
        )
        y_emb: Float[Tensor, "batch seq_len d_embed"] = self.pos_encoder(
            self.output_emb(y) * np.sqrt(self.dim_model)
        )
        y_hat: Float[Tensor, "batch seq_len d_embed"] = super().forward(
            src=x_emb,
            tgt=y_emb,
            src_mask=x_mask,
            tgt_mask=y_mask,
            src_key_padding_mask=x_pad_mask,
            tgt_key_padding_mask=y_pad_mask,
            memory_key_padding_mask=x_pad_mask,
            tgt_is_causal=True,
        )

        return y_hat

    def forward(
        self,
        x: Int[Tensor, "batch seq"],
        y: Int[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch seq generator"]:
        """Runs forward training pass for this seq2seq transformer training.

        Parameters:
            x (Tensor): Input sequence to train.
            y (Tensor): Output sequence to train.

        Returns:
            Tensor: output logits.
        """
        y_hat = self._forward(x, y)

        y_hat_tokens: Float[Tensor, "batch seq generator"] = self.generator(y_hat)

        # y_hat will return predicted tokens of y[1:], so we'll
        # copy over the original SOS token.
        sos_one_hot = torch.zeros_like(y_hat_tokens[:, 0, :])
        sos_one_hot = sos_one_hot.scatter(1, y[:, 0].unsqueeze(0).T, 1).unsqueeze(1)

        y_hat_logits = F.log_softmax(y_hat_tokens, dim=-1)
        return torch.cat([sos_one_hot, y_hat_logits], dim=1)

    def encode(
        self,
        x: Float[Tensor, "batch seq_len"],
        x_mask: Optional[Bool[Tensor, "seq_len seq_len"]],
    ) -> Float[Tensor, "batch seq_len d_model"]:
        """Split encoder and decoder runs."""
        x_mask = x_mask if x_mask else self.create_pad_mask(x)

        x_emb: Float[Tensor, "batch seq_len d_embed"] = self.pos_encoder(
            self.input_emb(x) * np.sqrt(self.dim_model)
        )
        z: Float[Tensor, "batch seq_len d_model"] = self.encoder(
            x_emb, src_key_padding_mask=x_mask
        )
        return z

    def decode(
        self,
        y: Float[Tensor, "batch seq_len"],
        memory: Float[Tensor, "batch seq_len d_model"],
        tgt_mask=None,
        memory_key_padding_mask=None,
    ):
        """Run decoder stage. This is needed for different decoding strategies."""
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(y.size(1))

        y_padding_mask = self.create_pad_mask(y)

        y_emb: Float[Tensor, "batch seq_len d_embed"] = self.pos_encoder(
            self.output_emb(y) * np.sqrt(self.dim_model)
        )

        return self.decoder(
            tgt=y_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=y_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )


class TransformerLRScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        """Optimiser based on the original Transformer implementation."""
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(TransformerLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self._step_count
        arg1 = step_num**-0.5
        arg2 = step_num * (self.warmup_steps**-1.5)
        return [
            base_lr * self.d_model**-0.5 * min(arg1, arg2) for base_lr in self.base_lrs
        ]


class PyTorchTransformerModule(NMTModule):
    def __init__(
        self,
        vocabulary_size,
        n_warmup_steps: int = 4000,
        name="PyTorchTransformer",
        sequence_length=100,
        **kwargs,
    ):
        super().__init__(
            name=name,
            vocabulary_size=vocabulary_size,
            sequence_length=sequence_length,
            **kwargs,
        )
        self.n_warmup_steps = n_warmup_steps
        self.model = PyTorchTransformerModel(
            vocab_size=self.vocabulary_size,
            decoder_vocab_size=self.decoder_vocabulary_size,
            batch_first=True,
            max_seq_length=sequence_length,
            **kwargs,
        )

    def configure_optimizers(self) -> Optimizer:
        optimiser = AdamWScheduleFree(
            self.model.parameters(),
            lr=self.learning_rate,
            warmup_steps=self.n_warmup_steps,
        )
        # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10000, T_mult=2)
        # scheduler = TransformerLRScheduler(
        #    optimiser, self.model.dim_model, self.n_warmup_steps
        # )
        return optimiser
