import torch
import torch.nn as nn
from torch.autograd import Variable
from . import modules, Constants
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


class Encoder(nn.Module):
    """
    LSTM Encoder.
    """
    def __init__(self, opt, vocabulary_size):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        input_size = opt.d_word_vec

        super(Encoder, self).__init__()
        self.word_lut = nn.Embedding(vocabulary_size,
                                  opt.d_word_vec,
                                  padding_idx=Constants.PAD)
        self.rnn = nn.LSTM(input_size,
                        self.hidden_size,
                        num_layers=opt.layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden=None):
        """
        if input is a tuple:
            (list of sequences, list of sequence lengths)
        otherwise:
            list of sequences
        """
        if isinstance(input, tuple):
            x, x_lengths = input
            emb = pack(self.word_lut(x), x_lengths)
        else:
            emb = self.word_lut(input)
        outputs, hidden_t = self.rnn(emb, hidden)

        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]
            
        return hidden_t, outputs

class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
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

class Decoder(nn.Module):

    def __init__(self, opt, vocabulary_size):
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.d_word_vec
        if self.input_feed:
            input_size += opt.rnn_size

        super(Decoder, self).__init__()
        self.word_lut = nn.Embedding(vocabulary_size,
                                  opt.d_word_vec,
                                  padding_idx=Constants.PAD)
        self.rnn = StackedLSTM(
            opt.layers,
            input_size,
            opt.rnn_size,
            opt.dropout)

        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions

        # self.rnn = nn.LSTM(input_size,
        #                 self.hidden_size,
        #                 num_layers=opt.layers,
        #                 dropout=opt.dropout,
        #                 bidirectional=opt.brnn)

        self.attn = modules.GlobalAttention(opt.rnn_size)
        self.dropout = nn.Dropout(opt.dropout)

        self.hidden_size = opt.rnn_size

        self.generator = nn.Sequential(
            nn.Linear(opt.rnn_size, vocabulary_size),
            nn.LogSoftmax(dim=1)
            )

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    # def forward(self, input, hidden, context, init_output):
    #     # print("INPUT FOR DECODER:", input.shape)
    #     emb = self.word_lut(input)
    #     #print(context.size())
    #     # n.b. you can increase performance if you compute W_ih * x for all
    #     # iterations in parallel, but that's only possible if
    #     # self.input_feed=False
    #     outputs = []
    #     output = init_output
    #     # print("DECODER EMB:", emb.shape)
    #     for i in range(input.shape[1]):
    #         # emb_t = emb[:,i,:]
    #         # emb_t = emb_t.unsqueeze(1)
    #         print("EMBT:", emb_t.shape)
    #         # emb_t = emb_t.squeeze(0)
    #         # if self.input_feed:
    #         #     emb_t = torch.cat([emb_t, output], 1)
    #         # print(hidden)
    #         # print(emb_t.shape)
    #         # print("HIDDEN:",hidden.shape)
    #         output, hidden = self.rnn(emb_t, hidden)
    #         print("OUT:",output.shape, context.shape)
    #         output, attn = self.attn(output.transpose(0, 1), context.transpose(0, 1))
    #         # output, attn = self.attn(output, context.transpose(0, 1))
    #         output = self.dropout(output)
    #         outputs += [output]

    #     outputs = torch.stack(outputs)
    #     return outputs, hidden, attn
    def forward(self, input, hidden, context, init_output, useGen=True):
        emb = self.word_lut(input)
        #print(context.size())
        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []
        output = init_output
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)

            output, hidden = self.rnn(emb_t, hidden)
            output, attn = self.attn(output, context.transpose(0, 1))
            output = self.dropout(output)
            if useGen:
                output = self.generator(output)
            outputs += [output]

        outputs = torch.stack(outputs)
        return outputs, hidden, attn

class NMTModel(nn.Module):

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def make_init_decoder_output(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def forward(self, src, tgt):
        x, x_length = src
        y, y_length = tgt
        # sort for pack_padded_sequences
        sorted_lengths, sorted_idx = torch.sort(x_length, descending=True)
        x = x[sorted_idx]
        y = y[sorted_idx]
        y_length = y_length[sorted_idx]
        # swap batch relationship order.
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)
        
        enc_hidden, context = self.encoder((x, sorted_lengths))
        init_output = self.make_init_decoder_output(context)

        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))

        out, dec_hidden, _attn = self.decoder(y, enc_hidden, context, init_output)
        # reverse tensor relationship order
        out = out.transpose(0, 1)
        # reverse order
        _, reversed_idx = torch.sort(sorted_idx)
        out = out[reversed_idx]

        return out
