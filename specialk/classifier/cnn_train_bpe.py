"""
CNN Training Runner.

Trains classifier used for style transfer.
"""

from __future__ import division

import specialk.classifier.onmt as onmt
from specialk.core.utils import log
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from tqdm import tqdm
from pathlib import Path
from typing import Union, Tuple, Optional
from torch.utils.data import DataLoader

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


DEVICE: str = onmt.core.check_torch_device()


def memory_efficient_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.modules.loss._Loss,
    eval=False,
) -> Tuple[int, torch.Tensor, int]:
    """Calculates loss between output and target against the given criterion.

    Args:
        outputs (torch.Tensor): Output generated from a model.
            dimensions: [batch_size, 1]
        targets (torch.Tensor): Target values the output should attempt to match.
            dimensions: [1, batch_size, 1]
        criterion (nn.modules.loss._Loss): Criterion metric to use.
        eval (bool, optional): Flag to check if metric is for eval only. If not set,
            then gradients are calculated. Defaults to False.

    Returns:
        _type_: _description_
    """
    # compute generations one piece at a time
    n_correct: int = 0
    loss: int = 0
    batch_size: int = outputs.size(0)

    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)

    loss_t = criterion(outputs.squeeze(-1), targets[0].float().squeeze(-1))
    n_correct = (outputs > 0.5).transpose(0, 1).long().eq(targets).sum()

    loss += loss_t.item()
    if not eval:
        # normalize the loss w.r.t batch size.
        loss_t.div(batch_size).backward()

    grad_output = None if outputs.grad is None else outputs.grad.data
    return loss, grad_output, n_correct


def calculate_metrics(
    output: torch.Tensor,
    target: torch.Tensor,
    loss: Optional[torch.Tensor] = None,
    criterion: Optional[nn.modules.loss._Loss] = None,
):
    """Calculate downstream metrics.

    This is not used for the loss.

    Args:
        output (torch.Tensor): Predicted values generated from the model.
        target (torch.Tensor): Values we want to predict.
        loss (Optional[torch.Tensor], optional): If set, then uses the loss 
            values instead. Defaults to None.
    """
    if not loss:
        loss = memory_efficient_loss(output, target, criterion, eval=True)

    n_correct: int = (output > 0.5).transpose(0, 1).long().eq(target).sum().item()
    num_words: int = target.size(1)
    # accuracy = 
    report_src_words += sum(batch[0][1])
    total_loss += loss
    total_n_correct += n_correct.item()
    total_words += num_words

    accuracy = (report_n_correct / report_tgt_words * 100) / opt.batch_size
    n_src_words_per_sec = (report_src_words / (time.time() - start)).item()

    


def eval(model, criterion, data, vocab_size: int, opt):
    total_loss = 0
    total_words = 0
    total_n_correct = 0

    model.eval()
    for batch in tqdm(data, desc="Eval"):
        src_seq, _, tgt_seq, _ = map(lambda x: x.to(DEVICE), batch)

        src = src_seq.transpose(0, 1)
        seq_len: int = src.size(0)
        batch_size: int = src.size(1)

        one_hot = Variable(torch.FloatTensor(seq_len, batch_size, vocab_size).zero_())
        one_hot_scatter = one_hot.scatter_(2, torch.unsqueeze(src, 2), 1)

        outputs = model(one_hot_scatter)
        targets = tgt_seq.transpose(0, 1)
        loss, _, n_correct = memory_efficient_loss(
            outputs, targets, criterion, eval=True
        )
        total_loss += loss
        total_n_correct += n_correct.item()
        total_words += targets.size(1)

    model.train()
    return total_loss / total_words, total_n_correct / total_words


def trainModel(
    model, data_train: DataLoader, data_validation: DataLoader, dataset, optim, opt
):
    model.train()

    # define criterion
    criterion = nn.BCELoss()

    # Vocab Size
    vocab_size = model.vocab_size

    start_time = time.time()

    def train_epoch(epoch: int, opt: dict):
        if opt.extra_shuffle and epoch > opt.curriculum:
            data_train.shuffle()

        # shuffle mini batch order
        batchOrder = torch.randperm(len(data_train))

        total_loss, total_words, total_n_correct = 0, 0, 0
        report_loss, report_tgt_words, report_src_words, report_n_correct = 0, 0, 0, 0
        start = time.time()
        i = 0
        for batch in tqdm(data_train, desc="Train"):
            # add tensors to memory
            src_seq, _, tgt_seq, _ = map(lambda x: x.to(DEVICE), batch)

            # batch = data_train[batchIdx][:-1] # exclude original indices

            # making one hot encoding
            src = src_seq.transpose(0, 1)

            inp = src  # Size is seq_len x batch_size, type: torch.cuda.LongTensor, Variable

            inp_ = torch.unsqueeze(
                inp, 2
            )  # Size is seq_len x batch_size x 1, type: torch.cuda.LongTensor, Variable

            one_hot = (
                torch.FloatTensor(src.size(0), src.size(1), vocab_size)
                .zero_()
                .to(DEVICE)
            )
            one_hot_scatt = one_hot.scatter_(
                2, inp_, 1
            )  # Size: seq_len x batch_size x vocab_size, type: torch.cuda.FloatTensor, Variable

            model.zero_grad()
            outputs = model(one_hot_scatt)

            targets = tgt_seq.transpose(0, 1)  # shape output to calculate loss.

            loss, gradients, n_correct = memory_efficient_loss(
                outputs, targets, criterion
            )
            outputs.backward(gradients)
            optim.step()  # update the parameters

            # metrics
            num_words = targets.size(1)
            report_loss += loss
            report_n_correct += n_correct.item()
            report_tgt_words += num_words
            report_src_words += sum(batch[0][1])
            total_loss += loss
            total_n_correct += n_correct.item()
            total_words += num_words
            runtime = time.time() - start_time
            accuracy = (report_n_correct / report_tgt_words * 100) / opt.batch_size
            n_src_words_per_sec = (report_src_words / (time.time() - start)).item()
            log.info(
                "Metrics",
                epoch=epoch,
                loss=loss,
                accuracy=accuracy,
                n_src_words_per_sec=n_src_words_per_sec,
                time_elapsed=runtime,
            )
            report_loss = report_tgt_words = report_src_words = report_n_correct = 0
            i += 1
        return total_loss / total_words, total_n_correct / total_words

    epoch: int
    for epoch in tqdm(range(opt.start_epoch, opt.epochs + 1), desc="Epoch"):
        #  (1) train for one epoch on the training set
        train_loss, train_acc = train_epoch(epoch, opt)
        print("Train accuracy: %g" % (train_acc * 100))
        print("Train Loss: ", train_loss)

        #  (2) evaluate on the validation set
        valid_loss, valid_acc = eval(model, criterion, data_validation, vocab_size, opt)
        print("Validation accuracy: %g" % (valid_acc * 100))
        print("Validation Loss: ", valid_loss)

        #  (3) update the learning rate
        optim.updateLearningRate(valid_loss, epoch)

        # save checkpoint
        checkpoint_filename = "%s_acc_%.2f_loss_%.2f_e%d.pt" % (
            opt.save_model,
            100 * valid_acc,
            valid_loss,
            epoch,
        )
        save_checkpoint(model, checkpoint_filename, opt, epoch, optim, dataset)


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[Path, str],
    opt: Optional[argparse.Namespace] = None,
    epoch: Optional[int] = 0,
    optim: Optional[torch.optim.Optimizer] = None,
    dataset: dict = {},
):
    """Save model checkpoint to model_path.

    Args:
        model (nn.Module): Model to save.
        checkpoint_path (Union[Path, str]): filepath
            (including filename) of checkpoint object.
    """
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
        checkpoint_path,
    )


def init_dataloaders(
    data: dict, batch_size: int, n_workers: int = 8
) -> Tuple[DataLoader, DataLoader]:
    """Initialise DataLoaders for dataset.

    Note that onmt has their own dataset loader, but no need to use that if we could
    leverage existing PyTorch Dataloaders.

    Args:
        data (dict): object container containing dataset.
        batch_size (int): batch size for dataset iteration.
        n_workers (int, Optional): number of workers to operate on the dataloader.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation dataloaders.
    """
    src_word2idx = data["dicts"]["src"]  # it's the same as the tgt.

    DATASET_IS_BPE = "byte_pairs" in src_word2idx.keys()
    if DATASET_IS_BPE:
        log.info("BPE Tokenised input detected.")
        # we have BPE loaded (detection is a heuristic).
        src_byte_pairs = {x + "_": y for x, y in src_word2idx["byte_pairs"].items()}
        src_word2idx = {**src_byte_pairs, **src_word2idx["words"]}
    else:
        log.info("Space-Separated Tokenised input detected.")

    train_loader = DataLoader(
        TranslationDataset(
            src_word2idx=src_word2idx,
            tgt_word2idx=src_word2idx,
            src_insts=data["train"]["src"],
            tgt_insts=data["train"]["tgt"],
        ),
        num_workers=n_workers,
        batch_size=batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True,
    )

    valid_loader = DataLoader(
        TranslationDataset(
            src_word2idx=src_word2idx,
            tgt_word2idx=src_word2idx,
            src_insts=data["valid"]["src"],
            tgt_insts=data["valid"]["tgt"],
        ),
        num_workers=n_workers,
        batch_size=batch_size,
        collate_fn=paired_collate_fn,
    )

    # printing for sanity checks.
    show_first_n: int = 2
    log.debug(
        f"Showing first {show_first_n} rows of training dataset.",
        source=data["train"]["src"][:show_first_n],
        target=data["train"]["tgt"][:show_first_n],
    )
    log.debug(
        f"Showing first {show_first_n} rows of validation dataset.",
        source=data["valid"]["src"][:show_first_n],
        target=data["valid"]["tgt"][:show_first_n],
    )

    return train_loader, valid_loader


def main():
    opt: argparse.Namespace = get_args()
    log.info("Loaded args", args=opt)

    log.info("Loading dataset dict.", dataset_path=opt.data)
    dataset: dict = torch.load(opt.data)

    dict_checkpoint = opt.train_from if opt.train_from else opt.train_from_state_dict
    if dict_checkpoint:
        log.info("Loading dicts from checkpoint at %s" % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint)
        dataset["dicts"] = checkpoint["dicts"]

    vocabulary_size: int
    if "settings" in dataset:
        vocabulary_size = dataset["dicts"]["src"]["kwargs"]["vocab_size"]
    else:
        vocabulary_size = dataset["dicts"]["src"].size()

    # create dataloaders
    data_train, data_validation = init_dataloaders(dataset, opt.batch_size)

    log.info(
        "Successfully loaded dataset.",
        dataset_path=opt.data,
        vocabulary_size=vocabulary_size,
        num_training_sequences=len(dataset["train"]["src"]),
        num_validation_sequences=len(dataset["valid"]["src"]),
        batch_size=opt.batch_size,
    )

    log.info("Building model...")

    model: onmt.CNNModels.ConvNet = onmt.CNNModels.ConvNet(opt, vocabulary_size)

    if opt.train_from:
        log.info("Loading model from checkpoint at %s" % opt.train_from)
        chk_model = checkpoint["model"]
        model_state_dict = {
            k: v for k, v in chk_model.state_dict().items() if "generator" not in k
        }
        model.load_state_dict(model_state_dict)
        opt.start_epoch = checkpoint["epoch"] + 1

    if opt.train_from_state_dict:
        log.info("Loading model from checkpoint at %s" % opt.train_from_state_dict)
        model.load_state_dict(checkpoint["model"])
        opt.start_epoch = checkpoint["epoch"] + 1

    if not opt.train_from_state_dict and not opt.train_from:
        # kaiming uniform
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

        model.load_pretrained_vectors(opt)

        optim: torch.optim.Optimizer = onmt.Optim(
            opt.optim,
            opt.learning_rate,
            opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
        )
    else:
        log.info("Loading optimizer from checkpoint:")
        optim = checkpoint["optim"]
        log.info(optim)

    model = model.to(DEVICE)

    if len(opt.gpus) > 1:
        model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)

    optim.set_parameters(model.parameters())

    if opt.train_from or opt.train_from_state_dict:
        optim.optimizer.load_state_dict(checkpoint["optim"].optimizer.state_dict())

    n_params = sum([p.nelement() for p in model.parameters()])
    log.info("Successfully initialised model weights.", n_params=n_params)

    trainModel(model, data_train, data_validation, dataset, optim, opt)


if __name__ == "__main__":
    main()
