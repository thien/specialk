"""
PyTorch native implementation of the Transformer.

Included to sanity check the remaining implementations.
This was based on this implementation:

https://github.com/pytorch/examples/blob/main/word_language_model/model.py
"""

import torch
from jaxtyping import Int, Float
from torch import Tensor, LongTensor
import torch.nn as nn
import torch.nn.functional as F
from specialk.models.mt_model import NMTModule
from specialk.models.transformer.Optim import ScheduledOptim
from specialk.models.transformer.pos_encoders import PositionalEncoder
import specialk.core.constants as Constants
from specialk.core.utils import log


class PyTorchTransformerModel(nn.Transformer):
    """PyTorch native implementation of a Transformer (see parent class)."""

    def __init__(
        self,
        vocab_size: int,
        dim_model: int,
        n_heads: int,
        dim_hidden: int,
        num_layers: int,
        dropout=0.5,
        max_seq_length=100,
        decoder_generator_weight_sharing=True,
        **kwargs,
    ):
        super(PyTorchTransformerModel, self).__init__(
            d_model=dim_model,
            nhead=n_heads,
            dim_feedforward=dim_hidden,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            **kwargs,
        )
        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.input_emb = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=dim_model
        )
        self.pos_encoder = PositionalEncoder(
            dim_model=dim_model, max_seq_length=max_seq_length
        )
        self.generator = nn.Linear(dim_model, vocab_size)
        self.tgt_mask = None
        self.decoder_generator_weight_sharing = decoder_generator_weight_sharing
        self.x_logit_scale = 1.0
        self.dropout = dropout
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz: int) -> LongTensor:
        """Generate square mask."""
        return torch.log(
            torch.tril(torch.ones(sz, sz, device=self.generator.weight.device))
        )

    def create_pad_mask(self, x: Tensor, pad_token: int) -> Tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return x == pad_token

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.generator.bias)
        nn.init.uniform_(self.generator.weight, -initrange, initrange)

    def forward(
        self, x: Int[Tensor, "batch_size seq_len"], has_mask=True
    ) -> Float[Tensor, "batch_size seq_len generator"]:
        """_summary_

        Args:
            x (LongTensor): Input sequence to perform translation.
            has_mask (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        if has_mask:
            len_x = len(x)
            if self.tgt_mask is None or self.tgt_mask.size(0) != len_x:
                mask = self._generate_square_subsequent_mask(len_x)
                self.tgt_mask = mask
        else:
            self.tgt_mask = None

        x = self.input_emb(x) * torch.sqrt(torch.Tensor(self.dim_model))
        x = self.pos_encoder(x)  # [batch, sequence_length, emb_size]
        z = self.encoder(x, mask=self.tgt_mask)  # [batch, sequence_length, dim_model]
        memory = torch.zeros(
            z.shape, device=x.device
        )  # keeps track of the tokens for autogregression.
        memory[:, 0, :] = self.input_emb(torch.LongTensor([Constants.SOS]))
        y = self.decoder(z, memory)
        y = self.generator(y)
        return F.log_softmax(y, dim=-1)


class PyTorchTransformerModule(NMTModule):
    def __init__(self, n_warmup_steps: int = 4000, **kwargs):
        super().__init__(**kwargs)
        self.n_warmup_steps = n_warmup_steps
        self.model = PyTorchTransformerModel(
            vocab_size=self.vocabulary_size,
            batch_first=True,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return ScheduledOptim(
            optimizer=torch.optim.Adam(
                filter(lambda x: x.requires_grad, self.model.parameters()),
                betas=(0.9, 0.98),
                eps=1e-09,
                lr=0.01,
            ),
            d_model=self.model.encoder.d_model,
            n_warmup_steps=self.n_warmup_steps,
        )
