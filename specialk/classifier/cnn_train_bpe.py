from __future__ import division

import specialk.classifier.onmt as onmt
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import sys
from tqdm import tqdm
from pathlib import Path

sys.path.append("../")
from specialk.core.dataset import TranslationDataset, collate_fn, paired_collate_fn


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train.py")

    parser.add_argument(
        "-data", required=True, help="Path to the *-train.pt file from preprocess.py"
    )
    parser.add_argument(
        "-save_model",
        default="model",
        help="""Model filename (the model will be saved as
                        <save_model>_epochN_PPL.pt where PPL is the
                        validation perplexity""",
    )
    parser.add_argument(
        "-train_from_state_dict",
        default="",
        type=str,
        help="""If training from a checkpoint then this is the
                        path to the pretrained model's state_dict.""",
    )
    parser.add_argument(
        "-train_from",
        default="",
        type=str,
        help="""If training from a checkpoint then this is the
                        path to the pretrained model.""",
    )
    parser.add_argument(
        "-num_classes", default=2, type=int, help="""Number of classes"""
    )

    ## Model options

    parser.add_argument(
        "-word_vec_size", type=int, default=300, help="Word embedding sizes"
    )
    parser.add_argument("-filter_size", default=5, type=int, help="Size of CNN filters")
    parser.add_argument(
        "-num_filters", default=100, type=int, help="Number of CNN filters"
    )
    parser.add_argument(
        "-sequence_length", type=int, default=50, help="Length of max sentence."
    )

    ## Optimization options

    parser.add_argument("-batch_size", type=int, default=64, help="Maximum batch size")
    parser.add_argument(
        "-max_generator_batches",
        type=int,
        default=32,
        help="""Maximum batches of words in a sequence to run
                        the generator on in parallel. Higher is faster, but uses
                        more memory.""",
    )
    parser.add_argument(
        "-epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "-start_epoch", type=int, default=1, help="The epoch from which to start"
    )
    parser.add_argument(
        "-param_init",
        type=float,
        default=0.1,
        help="""Parameters are initialized over uniform distribution
                        with support (-param_init, param_init)""",
    )
    parser.add_argument(
        "-optim", default="sgd", help="Optimization method. [sgd|adagrad|adadelta|adam]"
    )
    parser.add_argument(
        "-max_grad_norm",
        type=float,
        default=5,
        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to max_grad_norm""",
    )
    parser.add_argument(
        "-dropout",
        type=float,
        default=0.2,
        help="Dropout probability; applied between LSTM stacks.",
    )
    parser.add_argument(
        "-curriculum",
        action="store_true",
        help="""For this many epochs, order the minibatches based
                        on source sequence length. Sometimes setting this to 1 will
                        increase convergence speed.""",
    )
    parser.add_argument(
        "-extra_shuffle",
        action="store_true",
        help="""By default only shuffle mini-batch order; when true,
                        shuffle and re-assign mini-batches""",
    )

    # learning rate
    parser.add_argument(
        "-learning_rate",
        type=float,
        default=1.0,
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

    # pretrained word vectors

    parser.add_argument(
        "-pre_word_vecs_enc",
        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the encoder side.
                        See README for specific formatting instructions.""",
    )

    # GPU
    parser.add_argument(
        "-gpus", default=[], nargs="+", type=int, help="Use CUDA on the listed devices."
    )

    parser.add_argument(
        "-log_interval", type=int, default=50, help="Print stats at this interval."
    )

    return parser.parse_args()


opt = get_args()

print(opt)

DEVICE=onmt.core.check_torch_device()

def NMTCriterion(vocabSize):
    crit = nn.BCELoss()
    if opt.gpus:
        crit.cuda()
    return crit


def memoryEfficientLoss(outputs, targets, generator, crit, eval=False):
    # compute generations one piece at a time
    num_correct, loss = 0, 0
    outputs = Variable(outputs.data, requires_grad=(not eval))

    # outputs and targets are size batch_size
    batch_size = outputs.size(0)
    loss_t = crit(outputs.squeeze(-1), targets[0].float().squeeze(-1))
    if opt.gpus:
        pred_t = torch.ge(
            outputs.data,
            torch.cuda.FloatTensor(outputs.size(0), outputs.size(1)).fill_(0.5),
        )
    else:
        pred_t = torch.ge(
            outputs.data, torch.FloatTensor(outputs.size(0), outputs.size(1)).to(DEVICE).fill_(0.5)
        )

    # print("OUTPUT:", (outputs >=0.5).transpose(0,1))
    # print("TARGET:", targets)
    # print("ACC:", (outputs >=0.5).transpose(0,1).long().eq(targets).sum())

    # w
    num_correct = pred_t.long().squeeze(-1).eq(targets[0].data).sum()
    num_correct = (outputs >= 0.5).transpose(0, 1).long().eq(targets).sum()
    # print("NUM COR:", num_correct)
    loss += loss_t.item()
    if not eval:
        loss_t.div(batch_size).backward()

    grad_output = None if outputs.grad is None else outputs.grad.data
    return loss, grad_output, num_correct


def eval(model, criterion, data, vocab_size, opt):
    total_loss = 0
    total_words = 0
    total_num_correct = 0

    model.eval()

    for batch in tqdm(data, desc="Eval"):
        src_seq, _, tgt_seq, _ = map(lambda x: x.to(DEVICE), batch)
        # batch = data[i][:-1] # exclude original indices

        # src = batch[0]
        # inp = src[0] % vocab_size # Size is seq_len x batch_size, type: torch.cuda.LongTensor, Variable
        # inp_ = torch.unsqueeze(inp, 2) # Size is seq_len x batch_size x 1, type: torch.cuda.LongTensor, Variable

        src = src_seq.transpose(0, 1)
        # tgt_seq= tgt_seq.squeeze(1)
        # print(src.shape, tgt_seq.shape)
        inp = src  # Size is seq_len x batch_size, type: torch.cuda.LongTensor, Variable
        # print(inp.shape)
        inp_ = torch.unsqueeze(inp, 2)

        tensor = torch.FloatTensor

        # print(src[0].shape)
        one_hot = Variable(tensor(src.size(0), src.size(1), vocab_size).zero_())
        one_hot_scatt = one_hot.scatter_(
            2, inp_, 1
        )  # Size: seq_len x batch_size x vocab_size, type: torch.cuda.FloatTensor, Variable

        outputs = model(one_hot_scatt)
        targets = tgt_seq.transpose(0, 1)
        loss, _, num_correct = memoryEfficientLoss(
            outputs, targets, model, criterion, eval=True
        )
        total_loss += loss
        total_num_correct += num_correct.item()
        total_words += targets.size(1)

    model.train()
    return total_loss / total_words, total_num_correct / total_words


def trainModel(model, trainData, validData, dataset, optim, opt):
    # print(model)
    sys.stdout.flush()
    model.train()

    # define criterion
    criterion = NMTCriterion(opt.num_classes)

    # Vocab Size
    vocab_size = model.vocab_size
    # vocab_size = dataset['dicts']['src'].size()

    start_time = time.time()

    def trainEpoch(epoch, opt):
        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))

        total_loss, total_words, total_num_correct = 0, 0, 0
        report_loss, report_tgt_words, report_src_words, report_num_correct = 0, 0, 0, 0
        start = time.time()
        i = 0
        for batch in tqdm(trainData, desc="Train"):
            src_seq, _, tgt_seq, _ = map(lambda x: x.to(DEVICE), batch)

            # batch = trainData[batchIdx][:-1] # exclude original indices

            # making one hot encoding
            src = src_seq.transpose(0, 1)
            # tgt_seq= tgt_seq.squeeze(1)
            # print(src.shape, tgt_seq.shape)
            inp = src  # Size is seq_len x batch_size, type: torch.cuda.LongTensor, Variable
            # print(inp.shape)
            inp_ = torch.unsqueeze(
                inp, 2
            )  # Size is seq_len x batch_size x 1, type: torch.cuda.LongTensor, Variable
            # print(inp_.shape)

            # tensor = torch.cuda.FloatTensor if len(opt.gpus) >= 1 else torch.FloatTensor


            # print(src[0].shape)
            one_hot = torch.FloatTensor(src.size(0), src.size(1), vocab_size).zero_().to(DEVICE)
            one_hot_scatt = one_hot.scatter_(
                2, inp_, 1
            )  # Size: seq_len x batch_size x vocab_size, type: torch.cuda.FloatTensor, Variable

            model.zero_grad()
            outputs = model(one_hot_scatt)
            # print("outputs:", outputs.shape)
            targets = tgt_seq.transpose(0, 1)
            # print("targets",targets.shape)
            loss, gradOutput, num_correct = memoryEfficientLoss(
                outputs, targets, model, criterion
            )
            outputs.backward(gradOutput)

            # update the parameters
            optim.step()
            num_words = targets.size(1)
            report_loss += loss
            report_num_correct += num_correct.item()
            report_tgt_words += num_words
            report_src_words += sum(batch[0][1])
            total_loss += loss
            total_num_correct += num_correct.item()
            total_words += num_words

            if i % opt.log_interval == -1 % opt.log_interval:
                runtime = time.time() - start_time

                print(
                    "Epoch %2d, %5d/%5d; acc: %6.2f;  %3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed"
                    % (
                        epoch,
                        i + 1,
                        len(trainData),
                        report_num_correct / report_tgt_words * 100,
                        report_src_words / (time.time() - start),
                        report_tgt_words / (time.time() - start),
                        runtime,
                    )
                )

                sys.stdout.flush()
                report_loss = report_tgt_words = report_src_words = (
                    report_num_correct
                ) = 0
                start = time.time()
            i += 1
        return total_loss / total_words, total_num_correct / total_words

    for epoch in tqdm(range(opt.start_epoch, opt.epochs + 1), desc="Epoch"):
        print("")

        #  (1) train for one epoch on the training set
        train_loss, train_acc = trainEpoch(epoch, opt)
        print("Train accuracy: %g" % (train_acc * 100))
        print("Train Loss: ", train_loss)

        #  (2) evaluate on the validation set
        valid_loss, valid_acc = eval(model, criterion, validData, vocab_size, opt)
        print("Validation accuracy: %g" % (valid_acc * 100))
        print("Validation Loss: ", valid_loss)

        sys.stdout.flush()
        #  (3) update the learning rate
        optim.updateLearningRate(valid_loss, epoch)
        model_state_dict = (
            model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
        )
        model_state_dict = {
            k: v for k, v in model_state_dict.items() if "generator" not in k
        }
        #  (4) drop a checkpoint
        checkpoint = {
            "model": model_state_dict,
            "dicts": dataset["dicts"],
            "opt": opt,
            "epoch": epoch,
            "optim": optim,
        }
        torch.save(
            checkpoint,
            "%s_acc_%.2f_loss_%.2f_e%d.pt"
            % (opt.save_model, 100 * valid_acc, valid_loss, epoch),
        )


def init_dataloaders(data, opt):
    src_word2idx = data["dicts"]["src"]
    # tgt_word2idx = data['dict']['tgt']

    if "__sow" in src_word2idx["byte_pairs"]:
        # we have BPE
        src_byte_pairs = {x + "_": y for x, y in src_word2idx["byte_pairs"].items()}
        # tgt_byte_pairs = {x+"_": y for x,y in tgt_word2idx['byte_pairs'].items()}
        src_word2idx = {**src_byte_pairs, **src_word2idx["words"]}
        # tgt_word2idx = {**tgt_byte_pairs, **tgt_word2idx['words']}

    train_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=src_word2idx,
            tgt_word2idx=src_word2idx,
            src_insts=data["train"]["src"],
            tgt_insts=data["train"]["tgt"],
        ),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=src_word2idx,
            tgt_word2idx=src_word2idx,
            src_insts=data["valid"]["src"],
            tgt_insts=data["valid"]["tgt"],
        ),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
    )

    return train_loader, valid_loader


def main():
    print("Loading data from '%s'" % opt.data)

    dataset = torch.load(opt.data)

    dict_checkpoint = opt.train_from if opt.train_from else opt.train_from_state_dict
    if dict_checkpoint:
        print("Loading dicts from checkpoint at %s" % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint)
        dataset["dicts"] = checkpoint["dicts"]

    trainData, validData = init_dataloaders(dataset, opt)

    vocabulary_size = 0

    if "settings" in dataset:
        vocabulary_size = dataset["dicts"]["src"]["kwargs"]["vocab_size"]
    else:
        vocabulary_size = dataset["dicts"]["src"].size()

    print(" * vocabulary size. source = %d;" % vocabulary_size)
    print(" * number of training sentences. %d" % len(dataset["train"]["src"]))
    print(" * maximum batch size. %d" % opt.batch_size)

    print("Building model...")

    model:onmt.CNNModels.ConvNet = onmt.CNNModels.ConvNet(opt, vocabulary_size)

    model.word_lut.to(DEVICE)

    if opt.train_from:
        print("Loading model from checkpoint at %s" % opt.train_from)
        chk_model = checkpoint["model"]
        model_state_dict = {
            k: v for k, v in chk_model.state_dict().items() if "generator" not in k
        }
        model.load_state_dict(model_state_dict)
        opt.start_epoch = checkpoint["epoch"] + 1

    if opt.train_from_state_dict:
        print("Loading model from checkpoint at %s" % opt.train_from_state_dict)
        model.load_state_dict(checkpoint["model"])
        opt.start_epoch = checkpoint["epoch"] + 1


    if not opt.train_from_state_dict and not opt.train_from:
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

        model.load_pretrained_vectors(opt)

        optim = onmt.Optim(
            opt.optim,
            opt.learning_rate,
            opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
        )
    else:
        print("Loading optimizer from checkpoint:")
        optim = checkpoint["optim"]
        print(optim)


    model = model.to(DEVICE)


    if len(opt.gpus) > 1:
        model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)


    optim.set_parameters(model.parameters())

    if opt.train_from or opt.train_from_state_dict:
        optim.optimizer.load_state_dict(checkpoint["optim"].optimizer.state_dict())

    nParams = sum([p.nelement() for p in model.parameters()])
    print("* number of parameters: %d" % nParams)

    trainModel(model, trainData, validData, dataset, optim, opt)


if __name__ == "__main__":
    main()
