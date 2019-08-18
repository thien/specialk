import argparse
from tqdm import tqdm
from lib.RecurrentModel import RecurrentModel as recurrent
from lib.TransformerModel import TransformerModel as transformer
import torch
import torch.nn as nn
import sys
from torch.autograd import Variable
import numpy as np

sys.path.append('classifier')
import classifier.onmt as onmt
import classifier.onmt.CNNModels as CNNModels

description = """
train_decoder.py

Trains the decoder for style-transfer specific purposes.
"""

def load_args():
    parser = argparse.ArgumentParser(description="train_decoder.py")
    parser.add_argument("-f")
    # data options
    parser.add_argument('-data', required=True,
                        help='path to the *.pt file that was computed through preprocess.py')

    parser.add_argument('-checkpoint_encoder', default="", type=str,
                        help="""
                        If training from a checkpoint, then this is the path to the pretrained encoder.
                        """)

    parser.add_argument('-checkpoint_decoder', default="", type=str,
                        help="""
                        If training from a checkpoint, then this is the path to the pretrained model.
                        """)
    
    parser.add_argument("-new_directory", action="store_true", help="""
                        If enabled, creates a new directory instead of using the directory where the encoder is loaded. (Assumes that the checkpoint_encoder flag is activated.)
                        """)
    
    parser.add_argument('-log', action='store_true',
                        help="""
                        Determines whether to enable logs, which will save status into text files.
                        """)

    parser.add_argument('-directory_name', type=str, default="",
                        help="""
                        Name of directory. If set, then it'll use that name instead.
                        Otherwise it'll generate one based on the timestamp.
                        """)

    parser.add_argument('-save_model', action='store_true',
                        help="""
                        Determines whether to save the model or not.
                        """)

    parser.add_argument('-save_mode', default='all', choices=['all', 'best'],
                        help="""
                        Determines whether to save all versions of the model or keep the best version.
                        """)
    
    parser.add_argument("-verbose", action="store_true", help="""
                        If enabled, prints messages to terminal.
                        """)
            

    # model options.
    parser.add_argument('-model', choices=['transformer', 'recurrent'], 
                        required=True, help="""
                        Either a recurrent (seq2seq model) or a transformer.
                        """)

    parser.add_argument('-cuda', action='store_true',
                        help="""
                        Determines whether to use CUDA or not. (You should.)
                        """)
    
    parser.add_argument('-multi_gpu', action='store_true',
                        help="""
                        Determines whether to use multiple GPUs.
                        """)

    parser.add_argument("-cuda_device", type=int, help="""
                        Determines which GPU to use for computation.
                        """)

    parser.add_argument('-batch_size', type=int, default=64, help="""
                        Determines batch size of input data, for feeding into the models.
                        """)

    parser.add_argument('-layers', type=int, default=6, help="""
                        Number of layers for the model. (Recommended to have 6 for Transformer, 2 for recurrent.)
                        """)

    parser.add_argument("-d_word_vec", type=int, default=300, help="""
                        Dimension size of the token vectors representing words (or characters, or bytes).
                        """)

    # training options
    parser.add_argument('-epochs', type=int, required=True, default=10, help="""
                        Number of epochs for training. (Note
                        that for transformers, the number of
                        sequences become considerably longer.)
                        """)

    parser.add_argument('-dropout', type=float, default=0.1, help="""
                        Dropout probability' applied between
                        self-attention layers/RNN Stacks.
                        """)

    # debugging options
    parser.add_argument('-telegram', type=str, default="", help="""
                        filepath to telegram API private key
                        and chatID to send messages to.
                        """)

    # transformer specific options
    parser.add_argument('-d_model', type=int, default=512, help="""
                        Dimension size of the model.
                        """)
    parser.add_argument('-d_inner_hid', type=int, default=2048, help="""
                        Dimension size of the hidden layers of the transformer.
                        """)
    parser.add_argument('-d_k', type=int, default=64, help="""
                        Key vector dimension size.
                        """)
    parser.add_argument('-d_v', type=int, default=64, help="""
                        Value vector dimension size.
                        """)
    parser.add_argument('-n_head', type=int, default=8, help="""
                        Number of attention heads.
                        """)
    parser.add_argument('-n_warmup_steps', type=int, default=4000, help="""
                        Number of warmup steps.
                        """)
    parser.add_argument('-embs_share_weight', action='store_true', help="""
                        If enabled, allows the embeddings of the encoder
                        and the decoder to share weights.
                        """)
    parser.add_argument('-proj_share_weight', action='store_true', help="""
                        If enabled, allows the projection/generator 
                        to share weights.
                        """)
    parser.add_argument('-label_smoothing', action='store_true', help="""
                        Enables label smoothing.
                        """)
    
    # RNN specific options
    parser.add_argument('-max_generator_batches', type=int, default=32, help="""
                        Maximum batches of words in a sequence to run
                        the generator on in parallel. Higher is faster, but uses
                        more memory.""")
    parser.add_argument('-input_feed', type=int, default=0, help="""
                        Feed the context vector at each time step as
                        additional input (via concatenation with the word
                        embeddings) to the decoder.""")
    parser.add_argument('-max_grad_norm', type=float, default=5, help="""
                        If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to max_grad_norm.
                        """)
    parser.add_argument('-curriculum', action="store_true",
                        help="""For this many epochs, order the minibatches based
                        on source sequence length. Sometimes setting this to 1 will
                        increase convergence speed.""")
    parser.add_argument('-brnn', action='store_true',
                        help='Use a bidirectional encoder')
    parser.add_argument('-brnn_merge', default='concat',
                        help="""Merge action for the bidirectional hidden states:
                        [concat|sum]""")
    parser.add_argument('-rnn_size', type=int, default=500,
                        help='Size of LSTM hidden states')

    #learning rate
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Starting learning rate. If adagrad/adadelta/adam is
                        used, then this is the global learning rate. Recommended
                        settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")
    parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set or (ii) epoch has gone past
                        start_decay_at""")
    parser.add_argument('-start_decay_at', type=int, default=8,
                        help="""Start decaying every epoch after and including this
                        epoch""")
    parser.add_argument('-optim', default='adam', 
                        choices=['sgd', 'adagrad', 'adadelta', 'adam'], 
                        help="Gradient optimisation method.")

    # classifier labels
    parser.add_argument("-label0", required=True, type=str, help="""
                        Label 0 for CNN classifier.
                        """)
    parser.add_argument("-label1", required=True, type=str, help="""
                        Label 1 for CNN classifier.
                        """)
    parser.add_argument("-label_target", required=True, type=str, help="""
                        Label target for CNN classifier.
                        """)
    
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

    classifier_data = torch.load(opt.classifier_model, map_location=lambda x, loc: x)
    class_opt = classifier_data['opt']
    class_model = CNNModels.ConvNet(class_opt, model.dataset_settings.vocab_size)
    class_model.max_sent_length = cnn_opt.max_sent_length 
    class_model.cuda()
    class_model.eval()

    return class_model

def BCELoss():
    return nn.BCELoss().cuda()

criterion2 = BCELoss()

def tf_forward(self, src_seq, src_pos, tgt_seq, tgt_pos):
    """
    Forward pass of the transformer.
    """
    tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]
    # compute encoder output
    enc_output, enc_slf_attn_list = self.encoder(src_seq, src_pos, True)
    # feed into decoder
    dec_out = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output.detach(), True)
    dec_output, dec_slf_attn_list, dec_enc_attn_list = dec_out
    return dec_output


def n_tokens_correct(pred, gold):
    """
    Calculates number of correct tokens.
    """
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(self.constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()
    return n_correct


def classifier_loss(self, pred, class_model, class_input, tgt_label):
    """
    Computes classifier loss.
    """

    is_cuda = len(opt.gpus) >= 1
    softmax = nn.Softmax(dim=1)
    softmax = softmax.cuda() if is_cuda else softmax

    # translate generator outputs to class inputs
    pred = Variable(pred.data, requires_grad=True, volatile=False)
    linear = class_input(pred)

    # setup class_tgt, cnn_padding
    tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
    seq_length = class_model.max_sent_length
    d0,d1,d2 = linear.size()

    cnn_padding = tensor(abs(seq_length - d0), d1, d2).zero_()
    # Create a batch_size long tensor filled with the label to be generated
    class_tgt = tensor(d1).fill_(tgt_label)

    # reshape it for softmax
    linear_mod = linear.view(-1, d2)
    soft_out = softmax(linear_mod)

    # shape it back s.t we can concatenate it with our zero box
    soft_out = soft_out.view(linear.size(0), linear.size(1), linear.size(2))

    # concatenate padding
    if linear.size(0) < seq_length:
        soft_cat = torch.cat((soft_out, cnn_padding.detach()), 0)
    else:
        soft_cat = soft_out[:seq_length]

    # get class prediction with cnn
    class_outputs = class_model(soft_cat).squeeze()

    # return loss
    return criterion2(class_outputs, Variable(class_tgt))


def performance(self, generator, pred_before, gold, smoothing=False):
    """
    Calculates token level accuracy.
    Smoothing can be applied if needed.
    """
    pred = generator(pred_before) * self.model.x_logit_scale

    # compute adversarial loss
    l_classifier = classifier_loss(self, pred_before)

    # compute reconstruction loss
    pred = pred.view(-1, pred.size(2))
    l_reconstruction = self.calculate_loss(pred, gold, smoothing)

    # count number of correct tokens (for stats) 
    n_correct = n_tokens_correct(pred, gold)
    # compute overall loss
    overall_loss = l_reconstruction * l_classifier

    return overall_loss, n_correct


def compute_epoch(self, dataset, validation=False):
    do_smoothing = self.opt.label_smoothing
    ep_losses = []

    i = 0
    for batch in tqdm(dataset):
        i += 1
        self.model.zero_grad()
        src_seq, src_pos, tgt_seq, tgt_pos = map(
            lambda x: x.to(self.device), batch)
        # setup gold variable
        gold = tgt_seq[:, 1:]
        pred = tf_forward(self.model, src_seq, src_pos, tgt_seq, tgt_pos)
        # compute performance
        loss, n_correct = performance(self,pred, gold, smoothing=do_smoothing)

if __name__ == "__main__":
    opt = load_args()
    opt = opt.parse_args()
    assert opt.epochs > 0

    model = transformer(opt)

    # load dataset
    model.load_dataset()

    # load model encoder and decoder
    model.load(opt.checkpoint_encoder, opt.checkpoint_decoder)
    model.setup_optimiser()

    # we're only trying to train the decoder and generator.
    model.model.encoder.eval()

    # load model classifier
    classifier = load_classifier(opt)

    # initiate a new generator for style specific purposes
    model_dim = model.model.decoder.layer_stack[0].slf_attn.w_qs.out_features

    generator = nn.Sequential(
        nn.Linear(model.model.generator.in_features, model.model.generator.out_features),
        nn.LogSoftmax(dim=1)).cuda()

    # learn NN that feeds the decoder output into the classifier
    class_input = nn.Sequential(
        nn.Linear(model_dim,
            classifier.word_lut.weight.shape[0])).cuda()

    criterion_2 = BCELoss()


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