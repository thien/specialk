import random
from argparse import Namespace
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from specialk.core import constants as Constants
from specialk.core.utils import log
from specialk.models.recurrent.GlobalAttention import GlobalAttention


class Encoder(nn.Module):
    """
    LSTM Encoder.
    """

    def __init__(self, opt: Namespace, vocabulary_size: int):
        super(Encoder, self).__init__()

        self.layers: int = opt.layers
        self.num_directions: int = 2 if opt.brnn else 1
        self.hidden_size: int = opt.rnn_size // self.num_directions
        assert opt.rnn_size % self.num_directions == 0

        self.word_lut = nn.Embedding(
            vocabulary_size, opt.d_word_vec, padding_idx=Constants.PAD
        )

        self.rnn = nn.LSTM(
            opt.d_word_vec,
            self.hidden_size,
            num_layers=opt.layers,
            dropout=opt.dropout,
            bidirectional=opt.brnn,
        )

    def load_pretrained_vectors(self, opt: Namespace):
        """In case you want to use GloVe embeddings."""
        if opt.pre_word_vecs_enc:
            self.word_lut.weight.data.copy_(torch.load(opt.pre_word_vecs_enc))

    def forward(
        self,
        input: Union[Tuple[Tensor, Tensor], Tensor],
        hidden: Optional[Tensor] = None,
    ):
        """
        if input is a tuple:
            (list of sequences, list of sequence lengths)
        otherwise:
            list of sequences
        """
        if isinstance(input, tuple):
            x: Int[Tensor, "length batch"]
            x, x_lengths = input
            emb = pack(self.word_lut(x), x_lengths.cpu())
        else:
            emb = self.word_lut(input)

        outputs, (hidden_n, cell_n) = self.rnn(emb, hidden)

        if isinstance(input, tuple):
            outputs, _ = unpack(outputs)

        outputs: Float[Tensor, "batch length d*hidden_size"]
        hidden_n: Float[Tensor, "d*num_layers length hidden_size"]
        cell_n: Float[Tensor, "d*num_layers length hidden_size"]
        return outputs, (hidden_n, cell_n)


class StackedLegacyLSTM(nn.Module):
    def __init__(self, num_layers: int, input_size: int, rnn_size: int, dropout: float):
        super(StackedLegacyLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input: Tensor, hidden: Tensor):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class StackedLSTM(nn.Module):
    def __init__(
        self, num_layers: int, input_size: int, hidden_size: int, dropout: float = 0.0
    ):
        """_summary_

        Args:
            num_layers (int): _description_
            input_size (int): _description_
            hidden_size (int): _description_
            dropout (float, optional): _description_. Defaults to 0.0.
        """
        super(StackedLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.layers = nn.ModuleList(
            [
                nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
                for i in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        input: Float[Tensor, "batch input_size"],
        hidden: Tuple[
            Float[Tensor, "num_layers batch hidden"],
            Float[Tensor, "num_layers batch hidden"],
        ],
    ) -> Tuple[
        Float[Tensor, "batch input_size"],
        Tuple[
            Float[Tensor, "num_layers batch hidden"],
            Float[Tensor, "num_layers batch hidden"],
        ],
    ]:
        """Run Forward pass on the stacked LSTM.

        Args:
            input (Tensor): input tensor of shape (batch_size, input_size)
            hidden (Tuple[Tensor, Tensor]): previous hidden tensors from prev. pass:
                hidden[0] (prev_hidden) shape: (num_layers, batch_size, hidden_size)
                hidden[1] (prev_cell) shape: (num_layers, batch_size, hidden_size)

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor]]:
                input shape: (batch_size, hidden_size)
                (stacked_hidden, stacked_cell) shapes: both (
                  num_layers, batch_size, hidden_size)
        """
        prev_hidden, prev_cell = hidden
        new_hidden_states, new_cell_states = [], []

        layer: nn.LSTMCell
        layer_output: Float[Tensor, "batch hidden"]
        new_cell: Float[Tensor, "batch hidden"]
        for i, layer in enumerate(self.layers):
            layer_output, new_cell = layer(input, (prev_hidden[i], prev_cell[i]))
            input = layer_output

            if self.dropout and i < self.num_layers - 1:
                input = self.dropout(input)

            new_hidden_states.append(layer_output)
            new_cell_states.append(new_cell)

        stacked_hidden: Float[Tensor, "num_layers batch hidden"]
        stacked_hidden = torch.stack(new_hidden_states)

        stacked_cell: Float[Tensor, "num_layers batch hidden"]
        stacked_cell = torch.stack(new_cell_states)


        # input shape: (batch_size, hidden_size)
        # (stacked_hidden, stacked_cell) shapes: both (num_layers, batch_size, hidden_size)
        return input, (stacked_hidden, stacked_cell)

    def init_hidden(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Initialise start hidden values.

        Args:
            batch_size (int): batch size to match input.

        Returns:
            Tuple[Tensor, Tensor]: Dummy hidden and cell tensors. 
        """
        device = self.layers[0].weight_ih.device
        return (
            torch.zeros(
                (self.num_layers, batch_size, self.hidden_size),
                device=device,
                requires_grad=False,
            ),
            torch.zeros(
                (self.num_layers, batch_size, self.hidden_size),
                device=device,
                requires_grad=False,
            ),
        )


class Decoder(nn.Module):
    def __init__(self, opt, vocabulary_size: int):
        super(Decoder, self).__init__()
        self.layers: int = opt.layers
        self.input_feed: bool = opt.input_feed
        self.input_size: int = opt.d_word_vec
        if self.input_feed:
            self.input_size += opt.rnn_size
        self.dim_embedding: int = opt.d_word_vec
        self.dim_model: int = opt.rnn_size
        self.p_dropout: float = opt.dropout

        self.word_lut = nn.Embedding(
            vocabulary_size, self.dim_embedding, padding_idx=Constants.PAD
        )
        self.rnn = StackedLSTM(
            self.layers, self.input_size, self.dim_model, self.p_dropout
        )

        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions

        self.attention = GlobalAttention(opt.rnn_size)
        self.dropout = nn.Dropout(self.p_dropout)

        self.generator = nn.Sequential(
            nn.Linear(opt.rnn_size, vocabulary_size), nn.LogSoftmax(dim=-1)
        )

        self.teacher_forcing_ratio = 0.9

    def load_pretrained_vectors(self, opt: Namespace):
        """In case you have pre-trained word2vec embeddings."""
        if opt.pre_word_vecs_dec:
            self.word_lut.weight.data.copy_(torch.load(opt.pre_word_vecs_dec))

    def forward(
        self,
        input: Float[Tensor, "batch seq_len"],
        hidden: Tuple[
            Float[Tensor, "d*n_layers batch hidden_size"],
            Float[Tensor, "d*n_layers batch hidden_size"],
        ],
        context: Float[Tensor, "n_layers length d*hidden_size"],
        init_output: Float[Tensor, "batch d_model"],
        use_gen: bool = True,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor]:
        """_summary_

        Args:
            input (Float[Tensor, "batch seq_len"]): _description_
            hidden (Float[Tensor, &quot;&quot;]): _description_
            context (Float[Tensor, &quot;&quot;]): _description_
            init_output (Float[Tensor, &quot;&quot;]): Initial output (usually zeros)
            use_gen (bool, optional): If set, use generator layer. Defaults to True.

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor], Tensor]: Tuple containing
                output:
                hidden:
                attn: Attention scores from eac
        """

        emb: Float[Tensor, "seq_len, batch d_emb"] = self.word_lut(input)
        seq_len, batch_size, d_emb = emb.shape
        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False

        # TODO: add flag to determine whether to store attention scores or not.
        outputs, attention_scores = [], []  # store outputs @ each time step.
        output = init_output  # Initialize output with init_output

        for i in range(seq_len):
            if i == 0 or (random.random() < self.teacher_forcing_ratio):
                # teacher forcing.
                emb_t = emb[i, :, :]
            else:
                # Use the model's previous output
                prev_output = outputs[-1] if use_gen else self.generator(outputs[-1])
                prev_output = prev_output.argmax(dim=-1)
                emb_t = self.word_lut(prev_output)

            if self.input_feed:
                emb_t: Float[Tensor, "batch 2*d_emb"]
                emb_t = torch.cat([emb_t, output], 1)

            output, hidden = self.rnn(emb_t, hidden)
            output, attention = self.attention(output, context)
            output = self.dropout(output)
            if use_gen:
                output = self.generator(output)
            outputs += [output]
            attention_scores += [attention]

        outputs = torch.stack(outputs)  # this is faster than catting tensors.

        attention_scores: Float[Tensor, "seq_len batch d_context"]
        attention_scores = torch.stack(attention_scores)

        return outputs, hidden, attention_scores


class NMTModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder

    def make_init_decoder_output(self, batch_size: int) -> Tensor:
        """Used for inference."""
        return torch.zeros(
            (batch_size, self.decoder.hidden_size),
            device=self.decoder.word_lut.weight.device,
            requires_grad=False,
        )

    def update_teacher_forcing_ratio(self, epoch: int, total_epochs: int):
        # Linearly decrease the teacher forcing ratio from 1.0 to 0.5 over the course of training
        self.decoder.teacher_forcing_ratio = max(0.5, 0.9 - (epoch / total_epochs))

    def _fix_enc_hidden(self, hidden: Tensor) -> Tensor:
        """
        Restructures the encoder output for the decoder
          as a hidden state, if a bidirectional LSTM is used.

        Input:
            hidden: [layers, batch, dim]
        """
        layers, batch, dim = hidden.size()

        if self.encoder.num_directions == 2:
            num_layers = layers // 2
            reshaped = hidden.view(num_layers, 2, batch, dim)
            swapped = reshaped.transpose(1, 2)
            return swapped.reshape(num_layers, batch, dim * 2)
        return hidden

    def forward(
        self, src: Float[Tensor, "batch seq"], tgt: Float[Tensor, "batch seq"]
    ) -> Float[Tensor, "batch seq vocab_size"]:
        """
        Forward pass of RNN training. This includes teacher
        training so this cannot be used for inference.
        """

        if isinstance(src, Tuple) and isinstance(tgt, Tuple):
            x, x_length = src
            y, y_length = tgt
        else:
            x, y = src, tgt

        # swap batch relationship order.
        x, y = x.transpose(0, 1), y.transpose(0, 1)

        context: Float[Tensor, "batch length d*hidden"]
        hddn_n: Float[Tensor, "d*num_layers length hidden"]
        cell_n: Float[Tensor, "d*num_layers length hidden"]

        context, (hddn_n, cell_n) = self.encoder(x)

        init_output: Float[Tensor, "batch hidden"]
        init_output = self.make_init_decoder_output(x.shape[0])

        hidden = (
            self._fix_enc_hidden(hddn_n),
            self._fix_enc_hidden(cell_n),
        )

        out, _, _ = self.decoder(y, hidden, context, init_output)
        # reverse tensor order (for batch)
        out = out.transpose(0, 1)

        # resulting tensor may not be contiguous because of all the cats.
        return out.contiguous()
