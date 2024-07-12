import torch
import torch.nn as nn

from specialk.core.constants import PAD
from typing import Optional, Dict
from specialk.core.utils import log


class ConvNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        num_filters: int = 100,
        filter_size: int = 5,
        dropout: int = 0.2,
        num_classes: int = 2,
        word_vec_size: int = 300,
        args: Optional[Dict] = None,
        **kwargs,
    ):
        """Initiate ConvNet. This architecture is optimised for sequence classification.

        Args:
            vocab_size (int): Vocabulary size of token. This is based on the tokenizer.
            num_filters (int, optional): Number of Filters to use. Defaults to 100.
            filter_size (int, optional): Kernel size for convolution. Defaults to 5.
            sequence_length (int, optional): Maximum sequence length. Defaults to 50.
            dropout (int, optional): Probability rate of dropout. Defaults to 0.2.
            num_classes (int, optional): Number of classes to predict. Defaults to 2.
            word_vec_size (int, optional): Vector dimension for each token. Defaults to 300.
            args (Optional[Dict], optional): Additional arguments to save (as a sanity check). Defaults to None.
        """
        super(ConvNet, self).__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.num_classes = num_classes
        self.word_vec_size = word_vec_size
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.pooling_window_size = self.sequence_length - self.filter_size + 1
        self.strides = (1, 1)
        self.debug = False
        self.shown_shape = False

        self.word_lut = nn.Embedding(
            self.vocab_size, self.word_vec_size, padding_idx=PAD
        )

        self.conv = nn.Conv2d(
            in_channels=self.word_vec_size,
            out_channels=self.num_filters,
            kernel_size=(self.filter_size, 1),
            stride=self.strides,
            bias=True,
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, self.pooling_window_size, 1), stride=(1, 1, 1)
        )
        self.dropout = nn.Dropout(self.dropout)
        self.linear = nn.Linear(self.num_filters, self.num_classes - 1)
        self.sigmoid = nn.Sigmoid()

    def load_pretrained_vectors(self, opt) -> None:
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        ## src size is seq_size x batch_size x vocab_size. Most cases (50 x 64 x v)
        seq_len, batch_size, vocab_size = input.shape

        ## matrix multiply instead of lookup
        emb = torch.mm(input.view(-1, vocab_size), self.word_lut.weight)
        # emb = torch.mm(input.view(-1, input.size(2)), self.word_lut.weight)
        emb = emb.view(-1, batch_size, self.word_vec_size)
        emb = emb.transpose(0, 1)
        emb = emb.transpose(1, 2)
        emb = emb.unsqueeze(-1)
        h_conv = self.conv(emb)
        h_relu = self.relu(h_conv)
        h_max = self.maxpool(h_relu)
        h_flat = h_max.view(-1, self.num_filters)
        h_drop = self.dropout(h_flat)
        lin_out = self.linear(h_drop)
        out = self.sigmoid(lin_out)
        if self.debug and (not self.shown_shape):
            log.debug("shape", input=input.shape)
            log.debug("shape", h_conv=h_conv.shape)
            log.debug("shape", h_relu=h_relu.shape)
            log.debug("shape", h_max=h_max.shape)
            log.debug("shape", h_flat=h_flat.shape)
            log.debug("shape", h_drop=h_drop.shape)
            log.debug("shape", lin_out=lin_out.shape)
            log.debug("shape", out=out.shape)
            self.shown_shape = True
        return out
