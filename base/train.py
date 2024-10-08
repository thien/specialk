"""
Deals with training the models.

train.py will load some datasets, and will produce some results and save the encoder and decoders seperately.
"""

import argparse

import torch
import torch.nn as nn
from lib.RecurrentModel import RecurrentModel as recurrent
from lib.TransformerModel import TransformerModel as transformer
from tqdm import tqdm


def load_args():
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument(
        "-f", help="Don't use this. it's not for anything. (For Notebooks)"
    )
    # data options
    parser.add_argument(
        "-data",
        required=True,
        help="path to the *.pt file that was computed through preprocess.py",
    )

    parser.add_argument(
        "-checkpoint_encoder",
        default="",
        type=str,
        help="""
                        If training from a checkpoint, then this is the path to the pretrained encoder.
                        """,
    )

    parser.add_argument(
        "-checkpoint_decoder",
        default="",
        type=str,
        help="""
                        If training from a checkpoint, then this is the path to the pretrained model.
                        """,
    )

    parser.add_argument(
        "-new_directory",
        action="store_true",
        help="""
                        If enabled, creates a new directory instead of using the directory where the encoder is loaded. (Assumes that the checkpoint_encoder flag is activated.)
                        """,
    )

    parser.add_argument(
        "-log",
        action="store_true",
        help="""
                        Determines whether to enable logs, which will save status into text files.
                        """,
    )

    parser.add_argument(
        "-directory_name",
        type=str,
        default="",
        help="""
                        Name of directory. If set, then it'll use that name instead.
                        Otherwise it'll generate one based on the timestamp.
                        """,
    )

    parser.add_argument(
        "-save_model",
        action="store_true",
        help="""
                        Determines whether to save the model or not.
                        """,
    )

    parser.add_argument(
        "-save_mode",
        default="all",
        choices=["all", "best"],
        help="""
                        Determines whether to save all versions of the model or keep the best version.
                        """,
    )

    parser.add_argument(
        "-verbose",
        action="store_true",
        help="""
                        If enabled, prints messages to terminal.
                        """,
    )

    # model options.
    parser.add_argument(
        "-model",
        choices=["transformer", "recurrent"],
        required=True,
        help="""
                        Either a recurrent (seq2seq model) or a transformer.
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
        "-multi_gpu",
        action="store_true",
        help="""
                        Determines whether to use multiple GPUs.
                        """,
    )

    parser.add_argument(
        "-cuda_device",
        type=int,
        help="""
                        Determines which GPU to use for computation.
                        """,
    )

    parser.add_argument(
        "-batch_size",
        type=int,
        default=64,
        help="""
                        Determines batch size of input data, for feeding into the models.
                        """,
    )

    parser.add_argument(
        "-layers",
        type=int,
        default=6,
        help="""
                        Number of layers for the model. (Recommended to have 6 for Transformer, 2 for recurrent.)
                        """,
    )

    parser.add_argument(
        "-d_word_vec",
        type=int,
        default=300,
        help="""
                        Dimension size of the token vectors representing words (or characters, or bytes).
                        """,
    )

    # training options
    parser.add_argument(
        "-epochs",
        type=int,
        required=True,
        default=10,
        help="""
                        Number of epochs for training. (Note
                        that for transformers, the number of
                        sequences become considerably longer.)
                        """,
    )

    parser.add_argument(
        "-dropout",
        type=float,
        default=0.1,
        help="""
                        Dropout probability' applied between
                        self-attention layers/RNN Stacks.
                        """,
    )

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

    # transformer specific options
    parser.add_argument(
        "-d_model",
        type=int,
        default=512,
        help="""
                        Dimension size of the model.
                        """,
    )
    parser.add_argument(
        "-d_inner_hid",
        type=int,
        default=2048,
        help="""
                        Dimension size of the hidden layers of the transformer.
                        """,
    )
    parser.add_argument(
        "-d_k",
        type=int,
        default=64,
        help="""
                        Key vector dimension size.
                        """,
    )
    parser.add_argument(
        "-d_v",
        type=int,
        default=64,
        help="""
                        Value vector dimension size.
                        """,
    )
    parser.add_argument(
        "-n_head",
        type=int,
        default=8,
        help="""
                        Number of attention heads.
                        """,
    )
    parser.add_argument(
        "-n_warmup_steps",
        type=int,
        default=4000,
        help="""
                        Number of warmup steps.
                        """,
    )
    parser.add_argument(
        "-embs_share_weight",
        action="store_true",
        help="""
                        If enabled, allows the embeddings of the encoder
                        and the decoder to share weights.
                        """,
    )
    parser.add_argument(
        "-proj_share_weight",
        action="store_true",
        help="""
                        If enabled, allows the projection/generator 
                        to share weights.
                        """,
    )
    parser.add_argument(
        "-label_smoothing",
        action="store_true",
        help="""
                        Enables label smoothing.
                        """,
    )

    # RNN specific options
    parser.add_argument(
        "-max_generator_batches",
        type=int,
        default=32,
        help="""
                        Maximum batches of words in a sequence to run
                        the generator on in parallel. Higher is faster, but uses
                        more memory.""",
    )
    parser.add_argument(
        "-input_feed",
        type=int,
        default=0,
        help="""
                        Feed the context vector at each time step as
                        additional input (via concatenation with the word
                        embeddings) to the decoder.""",
    )
    parser.add_argument(
        "-max_grad_norm",
        type=float,
        default=5,
        help="""
                        If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to max_grad_norm.
                        """,
    )
    parser.add_argument(
        "-curriculum",
        action="store_true",
        help="""For this many epochs, order the minibatches based
                        on source sequence length. Sometimes setting this to 1 will
                        increase convergence speed.""",
    )
    parser.add_argument(
        "-brnn", action="store_true", help="Use a bidirectional encoder"
    )
    parser.add_argument(
        "-brnn_merge",
        default="concat",
        help="""Merge action for the bidirectional hidden states:
                        [concat|sum]""",
    )
    parser.add_argument(
        "-rnn_size", type=int, default=500, help="Size of LSTM hidden states"
    )

    # learning rate
    parser.add_argument(
        "-learning_rate",
        type=float,
        default=0.001,
        help="""Starting learning rate. If adagrad/adadelta/adam is
                        used, then this is the global learning rate. Recommended
                        settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""",
    )
    parser.add_argument(
        "-learning_rate_decay",
        type=float,
        default=0.5,
        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set or (ii) epoch has gone past
                        start_decay_at""",
    )
    parser.add_argument(
        "-start_decay_at",
        type=int,
        default=8,
        help="""Start decaying every epoch after and including this
                        epoch""",
    )
    parser.add_argument(
        "-optim",
        default="adam",
        choices=["sgd", "adagrad", "adadelta", "adam"],
        help="Gradient optimisation method.",
    )

    # style transfer specific features
    parser.add_argument(
        "-freeze_encoder",
        action="store_true",
        help="""
                        If enabled, freezes learning of the encoder.
                        """,
    )
    parser.add_argument(
        "-freeze_decoder",
        action="store_true",
        help="""
                        If enabled, freezes learning of the decoder.
                        """,
    )

    return parser


def train_model(opt):
    model = transformer(opt) if opt.model == "transformer" else recurrent(opt)
    print("Setup model wrapper.")
    model.load_dataset()
    print("Loaded data.")

    if opt.checkpoint_encoder:
        model.load(opt.checkpoint_encoder, opt.checkpoint_decoder)
        if opt.checkpoint_decoder:
            print("Loaded model encoder and decoder.")
        else:
            print("Loaded model encoder.")
    else:
        model.initiate()
        print("Initiated model and weights.")

    model.setup_optimiser()
    print("Training model.")
    if model.opt.save_model:
        model.init_logs()

    if torch.cuda.device_count() > 1 and model.opt.multi_gpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model.model.encoder = nn.DataParallel(model.model.encoder)
        model.model.decoder = nn.DataParallel(model.model.decoder)

    for epoch in tqdm(
        range(1, model.opt.epochs + 1), desc="Epochs", dynamic_ncols=True
    ):
        model.opt.current_epoch = epoch
        stats = model.train(epoch)
        if model.opt.save_model:
            model.save(epoch=epoch, note="epoch_" + str(epoch))
            model.update_logs(epoch)
    print("Done.")
    return model


if __name__ == "__main__":
    opt = load_args()
    opt = opt.parse_args()
    # validation.
    assert opt.epochs > 0

    train_model(opt)
