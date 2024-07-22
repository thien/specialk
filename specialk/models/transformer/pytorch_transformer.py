"""
PyTorch native implementation of the Transformer.

Included to sanity check the remaining implementations.
This was based on this implementation:

https://github.com/pytorch/examples/blob/main/word_language_model/model.py
https://pytorch.org/tutorials/beginner/translation_transformer.html which doesn't actually work.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import LongTensor, Tensor

import specialk.core.constants as Constants
from specialk.core.utils import log
from specialk.models.mt_model import NMTModule
from specialk.models.transformer.Optim import ScheduledOptim
from specialk.models.transformer.pos_encoders import PositionalEncoder


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

    def generate_square_subsequent_mask(self, size: int) -> LongTensor:
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
            Tensor: output tokens to
        """

        # create masks
        length = self.max_seq_length
        x_pad_mask = self.create_pad_mask(x)
        y_pad_mask = self.create_pad_mask(y)
        x_mask = torch.zeros((length, length), device=x.device).type(torch.bool)
        y_mask = self.generate_square_subsequent_mask(length)

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
        )
        y_hat_tokens: Float[Tensor, "batch seq generator"] = self.generator(y_hat)
        return F.log_softmax(y_hat_tokens, dim=-1)

    def encode(
        self, x: Float[Tensor, "batch seq_len"], x_mask: Bool[Tensor, "seq_len seq_len"]
    ) -> Float[Tensor, "batch seq_len d_model"]:
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
        y: Float[Tensor, "batch seq_len"],
        memory: Float[Tensor, "batch seq_len d_model"],
        y_mask: Int[Tensor, ""],
    ) -> Float[Tensor, "batch seq_len d_embed"]:
        """Run decoder stage. This is needed for different decoding strategies."""
        y_emb = self.output_emb(y)
        y_emb = self.pos_encoder(y_emb)
        return self.decoder(tgt=y_emb, memory=memory, tgt_mask=y_mask)


class PyTorchTransformerModule(NMTModule):
    def __init__(
        self,
        n_warmup_steps: int = 4000,
        name="PyTorchTransformer",
        vocabulary_size=35000,
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
