"""
Takes an NMT encoder and a style contingent decoder
to perform style transfer. It can also be used for
general machine translation. (i.e. it is contingent
on the encoder and decoders used.)
"""

import argparse

import torch
from lib.RecurrentModel import RecurrentModel as recurrent
from lib.TransformerModel import TransformerModel as transformer
from tqdm import tqdm


def load_args():
    parser = argparse.ArgumentParser(description="train.py")
    # data options

    parser.add_argument(
        "-model",
        choices=["transformer", "recurrent"],
        required=True,
        help="""
                        Either a recurrent (seq2seq model) or a transformer.
                        """,
    )

    parser.add_argument(
        "-src",
        default="",
        required=True,
        type=str,
        help="""
                        Source sequence for decoding purposes (one line per sequence).
                        """,
    )

    parser.add_argument(
        "-vocab",
        required=True,
        help="""
                        Vocabulary (refer to training data pickle.
                        """,
    )

    parser.add_argument(
        "-checkpoint_encoder",
        required=True,
        default=None,
        type=str,
        help="""
                        If training from a checkpoint, then this is the path to the pretrained encoder.
                        """,
    )

    parser.add_argument(
        "-checkpoint_decoder",
        required=True,
        default=None,
        type=str,
        help="""
                        If training from a checkpoint, then this is the path to the pretrained model.
                        """,
    )

    parser.add_argument(
        "-output",
        default="predictions.txt",
        help="""Path to output the predictions (each line will
                        be the decoded sequence""",
    )

    parser.add_argument(
        "-copysrc",
        action="store_true",
        help="""
                        If enabled, saves a copy of the source sequences into the model folder.
                        """,
    )

    parser.add_argument(
        "-cuda",
        action="store_true",
        help="""
                        Determines whether to use CUDA or not. (You should.)
                        """,
    )

    parser.add_argument(
        "-cuda_device",
        type=int,
        help="""
                        Determines which GPU to use for computation.
                        """,
    )

    # debugging options
    parser.add_argument(
        "-telegram_key",
        help="""
                        filepath to telegram API private key to send messages to.
                        """,
    )

    # translate option
    parser.add_argument(
        "-batch_size",
        type=int,
        default=128,
        help="""
                        Determines batch size of input data, for feeding into the models.
                        """,
    )
    parser.add_argument("-beam_size", type=int, default=5, help="Beam size")

    parser.add_argument(
        "-n_best",
        type=int,
        default=1,
        help="""If verbose is set, will output the n_best
                        decoded sentences""",
    )

    parser.add_argument("-verbose", action="store_true")

    # debugging options
    parser.add_argument(
        "-telegram",
        type=str,
        default="",
        help="""
                        filepath to telegram API private key
                        and chatID to send messages to.
                        """,
    )

    parser.add_argument(
        "-override_max_token_seq_len",
        type=int,
        help="""
                        If allocated, changes the max sequence length.
    """,
    )
    opt = parser.parse_args()

    opt.save_model = False
    opt.new_directory = False

    # validation.

    return opt


import torch

if __name__ == "__main__":
    opt = load_args()
    if opt.cuda_device:
        torch.cuda.set_device(opt.cuda_device)
    model = transformer(opt) if opt.model == "transformer" else recurrent(opt)

    print("Setup model wrapper.")
    model.load(opt.checkpoint_encoder, opt.checkpoint_decoder)
    print("Initiated model and weights.")

    # translate sequences
    hypotheses = model.translate()
