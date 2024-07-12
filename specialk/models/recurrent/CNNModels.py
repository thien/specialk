import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

import specialk.core.constants as Constants


class ConvNet(nn.Module):
    def __init__(self, opt, dicts):
        super(ConvNet, self).__init__()

        self.num_filters = opt.num_filters
        pooling_window_size = opt.sequence_length - opt.filter_size + 1
        self.strides = (1, 1)
        self.vocab_size = dicts.size()
        self.word_vec_size = opt.word_vec_size

        self.word_lut = nn.Embedding(
            self.vocab_size, opt.word_vec_size, padding_idx=onmt.Constants.PAD
        )

        self.conv1 = nn.Conv2d(
            in_channels=self.word_vec_size,
            out_channels=self.num_filters,
            kernel_size=(opt.filter_size, 1),
            stride=self.strides,
            bias=True,
        )

        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool3d(
            kernel_size=(1, pooling_window_size, 1), stride=(1, 1, 1)
        )
        self.dropout = nn.Dropout(opt.dropout)
        # self.linear = nn.Linear(opt.num_filters, opt.num_classes)
        self.linear = nn.Linear(opt.num_filters, opt.num_classes - 1)
        self.sigmoid = nn.Sigmoid()
        # self.logsoftmax = nn.LogSoftmax()

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input):
        ## src size is seq_size x batch_size x vocab_size. Most cases (50 x 64 x v)
        # emb = self.word_lut(src[0])
        # print("INP:",input.shape)
        emb = torch.mm(input.view(-1, input.size(2)), self.word_lut.weight)
        # print("EMB:", emb.shape)
        emb = emb.view(-1, input.size(1), self.word_vec_size)
        # print("EM1:", emb.shape)
        emb = emb.transpose(0, 1)
        emb = emb.transpose(1, 2)
        emb = emb.unsqueeze(-1)
        # print("EM2:", emb.shape)
        h_conv = self.conv1(emb)
        # print("CN1:", h_conv.shape)
        h_relu = self.relu1(h_conv)
        # print("RLU:", h_relu.shape)
        # @resize h_relu
        # if len(h_relu.shape) > 3:
        # 	h_relu = h_relu.squeeze(-1)
        # print("->", h_relu.shape)
        h_max = self.maxpool1(h_relu)
        h_flat = h_max.view(-1, self.num_filters)
        h_drop = self.dropout(h_flat)
        lin_out = self.linear(h_drop)
        out = self.sigmoid(lin_out)
        return out
