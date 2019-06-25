"""
Deals with training the models.

train.py will load some datasets, and will produce some results and save the encoder and decoders seperately.
"""

import argparse
import math
import time

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import core.constants as Constants

import lib.RecurrentModel as recurrent
import lib.TransformerModel as transformer

def load_args():
    parser = argparse.ArgumentParser(description="train.py")
    # data options
    parser.add_argument('-data', required=True,
                        help='path to the *.pt file that was computed through preprocess.py')
    parser.add_argument('-model_name', default="model",
                        help="""
                        Model filename (the model will be saved as <model_name>_epochN_PPL.pt where PPL is the validation perplexity.
                        """)
    parser.add_argument('-train_from_state_dict', default='',                          type=str, help="""
                        If training form a checkpoint, then this is the path to the pretrained model's state_dict.""")
    parser.add_argument('-checkpoint_encoder', default="", type=str,
                        help="""
                        If training from a checkpoint, then this is the path to the pretrained encoder.
                        """)
    parser.add_argument('-checkpoint_decoder', default="", type=str,
                        help="""
                        If training from a checkpoint, then this is the path to the pretrained model.
                        """)
    # model options.
    parser.add_argument('-model', required=True,
                        help="""
                        Either a recurrent (seq2seq model) or a transformer.
                        """)
    parser.add_argument('-cuda', default=True, type=bool, 
                        help='determines whether to use CUDA or not.')
    # training options
    parser.add_argument('-epochs', type=int, required=True, help="""
                        Number of epochs for training. (Note that for transformers, the number of sequences become considerably longer.)
                        """)
    # additional options
    parser.add_argument('-telegram_key', help="""
                        filepath to telegram API private key to send messages to.
                        """)

    opt = parser.parse_args()

    # validation.
    assert opt.model in ["transformer", "recurrent"]
    assert opt.epochs > 0

    return opt

def load_dataset(opt):
    data = torch.load(opt.data)

def init_model(opt):
    model = None
    if opt.model == "transformer":
        model = transformer(opt)
    else:
        model = recurrent(opt)
    model.init()

def train(dataset, model, args):
    for epoch in range(1, args.epochs +1):
        model.train(epoch)

if __name__ == "__main__":
    args = load_args()
    dataset = load_dataset(args)
    model = init_model(args)
    train(dataset, model, args)