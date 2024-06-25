import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from specialk.lib.RecurrentModel import RecurrentModel as recurrent
from lib.TransformerModel import TransformerModel as transformer
from torch.autograd import Variable
from tqdm import tqdm

sys.path.append("classifier")
import specialk.classifier.onmt as onmt
import specialk.classifier.onmt.CNNModels as CNNModels

description = """
train_decoder.py

Trains the transformer decoder for style-transfer specific purposes.
"""


def load_args():
    parser = argparse.ArgumentParser(description="train_decoder.py")
    parser.add_argument("-f")
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

    # classifier labels
    parser.add_argument(
        "-classifier_model",
        required=True,
        type=str,
        help="""
                        path for classifier model.
                        """,
    )
    parser.add_argument(
        "-label0",
        required=True,
        type=str,
        help="""
                        Label 0 for CNN classifier.
                        """,
    )
    parser.add_argument(
        "-label1",
        required=True,
        type=str,
        help="""
                        Label 1 for CNN classifier.
                        """,
    )
    parser.add_argument(
        "-label_target",
        required=True,
        type=str,
        help="""
                        Label target for CNN classifier.
                        """,
    )

    return parser


def load_classifier(opt):
    # load model classifier
    cnn_opt = argparse.Namespace()
    cnn_opt.model = opt.classifier_model
    # need to shorten the model sequences!
    cnn_opt.max_sent_length = model.opt.max_token_seq_len
    cnn_opt.label0 = opt.label0
    cnn_opt.label1 = opt.label1
    cnn_opt.tgt = opt.label_target
    cnn_opt.cuda = 0
    cnn_opt.batch_size = opt.batch_size
    cnn_opt.tgt_label = 1 if opt.label_target == opt.label1 else 0

    classifier_data = torch.load(opt.classifier_model, map_location=lambda x, loc: x)
    class_opt = classifier_data["opt"]
    class_model = CNNModels.ConvNet(class_opt, model.dataset_settings.vocab_size)
    class_model.max_sent_length = cnn_opt.max_sent_length
    class_model.cuda()
    class_model.eval()
    class_model.refs = cnn_opt

    return class_model


def BCELoss():
    return nn.BCELoss().cuda()


def tf_forward(self, src_seq, src_pos, tgt_seq, tgt_pos):
    """
    Forward pass of the transformer.
    """
    tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]
    # compute encoder output
    enc_output, _ = self.encoder(src_seq, src_pos, True)
    # feed into decoder
    dec_out = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output.detach(), True)
    dec_output, _, _ = dec_out
    return dec_output


def n_tokens_correct(pred, gold, pad_token):
    """
    Calculates number of correct tokens.
    """
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(pad_token)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()
    return n_correct


def classifier_loss(self, pred, class_model, class_input):
    """
    Computes classifier loss.
    """

    # need to fix is_cuda
    is_cuda = torch.cuda.device_count() > 0
    softmax = nn.Softmax(dim=1)
    softmax = softmax.cuda() if is_cuda else softmax
    tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
    seq_length = class_model.opt.sequence_length
    target_label = class_model.refs.tgt_label

    # translate generator outputs to class inputs
    exp = Variable(pred.data, requires_grad=True, volatile=False)
    linear = class_input(exp)
    d0, d1, d2 = linear.size()

    # reshape it for softmax, and shape it back.
    out = softmax(linear.view(-1, d2)).view(d0, d1, d2)

    # our dataset is current (batch, seqlen, vocab)
    # but CNN takes in (seqlen, batch, vocab)
    out = out.transpose(0, 1)
    d0, d1, d2 = out.size()

    # setup padding because the CNN only accepts inputs of certain dimension
    if d0 < seq_length:
        # pad sequences
        cnn_padding = tensor(abs(seq_length - d0), d1, d2).zero_()
        cat = torch.cat((out, cnn_padding), 0)
    else:
        # trim sequences
        cat = out[:seq_length]

    # get class prediction with cnn
    class_outputs = class_model(cat).squeeze()

    # return loss between predicted class outputs and a vector of class_targets
    return criterion2(class_outputs, tensor(d1).fill_(target_label))


def performance(self, pred_before, gold, class_input, class_model, smoothing=False):
    """
    Calculates token level accuracy.
    Smoothing can be applied if needed.
    """
    pred = self.model.generator(pred_before) * self.model.x_logit_scale

    # compute adversarial loss
    l_classifier = classifier_loss(self, pred_before, class_model, class_input)

    # compute reconstruction loss
    pred = pred.view(-1, pred.size(2))
    l_reconstruction = self.calculate_loss(pred, gold, smoothing)

    # count number of correct tokens (for stats)
    n_correct = n_tokens_correct(pred, gold, pad_token=self.constants.PAD)

    return l_reconstruction, l_classifier, n_correct


def compute_epoch(self, class_input, class_model, dataset, validation=False):
    if validation:
        self.model.decoder.eval()
        self.model.generator.eval()
    else:
        self.model.decoder.train()
        self.model.generator.train()

    smooth = self.opt.label_smoothing
    losses, recon, classes, accs = [], [], [], []
    recon = []

    i = 0
    for batch in tqdm(dataset, desc="Training", dynamic_ncols=True):
        i += 1
        self.model.zero_grad()
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(self.device), batch)
        # setup gold variable
        gold = tgt_seq[:, 1:]
        pred = tf_forward(self.model, src_seq, src_pos, tgt_seq, tgt_pos)
        # compute performance
        l_recon, l_class, n_correct = performance(
            self, pred, gold, class_input, class_model, smooth
        )
        net_loss = l_recon * l_class

        if not validation:
            # gradient descent
            net_loss.backward()
            # update parameters
            self.optimiser.step_and_update_lr()
        else:
            # generate outputs
            self.save_eval_outputs(
                self.model.generator(pred) * self.model.x_logit_scale,
                output_dir="eval_outputs_st_" + self.label_target,
            )

        # store results
        losses.append(net_loss.item())
        recon.append(l_recon.item())
        classes.append(l_class.item())
        accuracy = losses[-1] / gold.ne(self.constants.PAD).sum().item()
        accs.append(accuracy)

        if i % 100 == 0 and not validation:
            del src_seq, src_pos, tgt_seq, tgt_pos, batch
            del gold, pred, net_loss, n_correct, l_recon, l_class
            torch.cuda.empty_cache()

    return losses, recon, classes, accs


def means(l):
    return [round(np.mean(x), 3) for x in l]


def save(decoder, generator, epoch, settings, vocab, filepath):
    checkpoint_decoder = {
        "type": "transformer",
        "model": decoder.state_dict(),
        "generator": generator.state_dict(),
        "epoch": epoch,
        "settings": settings,
        "vocab": vocab,
    }

    if checkpoint_decoder["settings"].telegram:
        del checkpoint_decoder["settings"].telegram

    torch.save(checkpoint_decoder, filepath)


if __name__ == "__main__":
    opt = load_args()
    opt = opt.parse_args()
    opt.model = "transformer"
    # generator
    assert opt.epochs > 0

    model = transformer(opt)
    model.label_target = opt.label_target
    # load dataset
    print("LOADING DATASET:", opt.data)
    model.load_dataset()
    vocab = model.tgt_bpe.vocabs_to_dict(False)
    # load model encoder and decoder
    model.load(opt.checkpoint_encoder, opt.checkpoint_decoder)

    model.setup_optimiser()

    # we're only trying to train the decoder and generator.
    model.model.encoder.eval()

    # load model classifier
    classifier = load_classifier(opt)
    criterion2 = BCELoss()

    # initiate a new generator for style specific purposes
    model_dim = model.model.decoder.layer_stack[0].slf_attn.w_qs.out_features

    # learn NN that feeds the decoder output into the classifier
    class_input = nn.Sequential(
        nn.Linear(model_dim, classifier.word_lut.weight.shape[0])
    ).cuda()

    criterion_2 = BCELoss()

    torch.cuda.empty_cache()

    print("model.opt.directory:", model.opt.directory)

    for ep in tqdm(range(1, opt.epochs + 1), desc="Epoch", dynamic_ncols=True):
        # train
        model.opt.current_epoch = ep
        train_results = compute_epoch(
            model, class_input, classifier, model.training_data
        )
        losses, recon, classes, accs = means(train_results)
        print("Training Loss:", losses, "(", recon, classes, ")", accs, "%")

        filename = "decoder_" + opt.label_target + "_epoch_" + str(ep) + ".chkpt"
        filepath = os.path.join(model.opt.directory, filename)
        save(model.model.decoder, model.model.generator, ep, opt, vocab, filepath)

        # validate
        with torch.no_grad():
            valid_results = compute_epoch(
                model, class_input, classifier, model.validation_data, True
            )
            losses, recon, classes, accs = means(valid_results)
            print("Validation Loss:", losses, "(", recon, classes, ")", accs, "%")

        torch.cuda.empty_cache()

    # need to save the model.


# load the dataset

# load the transformer encoder model

# load the cnn classifier model

# load the decoder model

# set it to customised ont_models_decoderModel

# setup dataparallel

# train:
# two criterions
# one for dataset
# one for classification

# at each epoch:
# train epoch
# shuffle data?
# iterate through train data
# enoder(input)

# val epoch

# save model
