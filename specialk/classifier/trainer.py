"""
CNN Training Runner.

Trains classifier used for style transfer.
Modified version of the following code:
https://github.com/shrimai/Style-Transfer-Through-Back-Translation/blob/master/classifier/cnn_train.py.


TODO: Check this works on both BPE and word-separated tokenized datasets from the huggingface datasets.
"""

from __future__ import division

import argparse
from pathlib import Path
from typing import Optional, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import AdvancedProfiler
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss as Loss
from torch.utils.data import DataLoader
from tqdm import tqdm

import specialk.classifier.onmt as onmt
from datasets import Dataset, load_dataset
from specialk.classifier.onmt.CNNModels import ConvNet
from specialk.core.constants import LOGGING_DIR, LOGGING_PERF_NAME, PROJECT_DIR
from specialk.core.utils import check_torch_device, log, namespace_to_dict
from specialk.datasets.dataloaders import (
    init_classification_dataloaders as init_dataloaders,
)
from specialk.models.tokenizer import BPEVocabulary, Vocabulary, WordVocabulary

DEVICE: str = check_torch_device()


class CNNClassifier(pl.LightningModule):
    """CNN Classifier in lightning wrapper."""

    def __init__(self, name: str, vocabulary_size: int, sequence_length: int):
        super().__init__()
        self.name = name
        self.vocabulary_size = vocabulary_size
        self.sequence_length = sequence_length
        self.model = ConvNet(
            vocab_size=self.vocabulary_size, sequence_length=self.sequence_length
        )
        self.criterion = nn.BCELoss()

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Run Training step.

        Args:
            batch (dict): individual batch generated from the DataLoader.
            batch_idx (int): ID corresponding to the batch.

        Returns:
            torch.Tensor: Returns loss.
        """
        x, y = batch["text"], batch["label"]

        # x = x.squeeze(2, 1)

        seq_len: int = x.size(1)
        batch_size: int = x.size(0)

        if batch_size < 50:
            log.error("batch size should be greater than 50.")

        # wrap into one-hot encoding of tokens for activation.
        one_hot = torch.zeros(
            seq_len, batch_size, self.vocabulary_size, device=self.device
        ).scatter_(2, torch.unsqueeze(x.T, 2), 1)

        y_hat = self.model(one_hot).squeeze(-1)
        loss = self.criterion(y_hat, y.float())

        accuracy = self.calculate_classification_metrics(y_hat, y)
        self.log_dict(
            {"train_acc": accuracy, "batch_id": batch_idx, "train_loss": loss},
            batch_size=batch_size,
        )
        return loss

    def _shared_eval_step(
        self, batch: dict, batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run shared eval step.

        Args:
            batch (dict): individual batch generated from the DataLoader.
            batch_idx (int): ID corresponding to the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns loss, accuracy.
        """
        x, y = batch["text"], batch["label"]

        seq_len: int = x.size(1)
        batch_size: int = x.size(0)

        # wrap into one-hot encoding of tokens for activation.
        one_hot = torch.zeros(
            seq_len, batch_size, self.vocabulary_size, device=self.device
        ).scatter_(2, torch.unsqueeze(x.T, 2), 1)

        y_hat = self.model(one_hot).squeeze(-1)
        loss = self.criterion(y_hat, y.float())

        accuracy = self.calculate_classification_metrics(y_hat, y)
        return loss, accuracy

    def validation_step(self, batch: dict, batch_idx: int):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, batch_size=batch["text"].size(0))
        return metrics

    def test_step(self, batch: dict, batch_idx: int):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def predict_step(
        self, batch: dict, batch_idx: int, dataloader_idx=0
    ) -> torch.Tensor:
        x = batch["text"]
        y_hat = self.model(x)
        return y_hat.squeeze(-1)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=0.02)

    @staticmethod
    def calculate_classification_metrics(
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> int:
        """Calculate downstream metrics.

        Args:
            output (torch.Tensor): Predicted values generated from the model.
            target (torch.Tensor): Values we want to predict.

        Returns:
            int: Accuracy.
        """

        outputs = (output > 0.5).long()
        n_correct: int = outputs.eq(target).sum().item() / outputs.size(0)
        return n_correct


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


def memory_efficient_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: Loss,
    eval=False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates loss between output and target against the given criterion.

    TODO: you don't need Variable anymore.

    Args:
        outputs (torch.Tensor): Output generated from a model.
            dimensions: [batch_size, 1]
        targets (torch.Tensor): Target values the output should attempt to match.
            dimensions: [1, batch_size, 1]
        criterion (Loss): Criterion metric to use.
        eval (bool, Optional): Flag to check if metric is for eval only. If not set,
            then gradients are calculated. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: loss tensor, gradients tensor.
    """
    # compute generations one piece at a time
    loss: int = 0
    batch_size: int = outputs.size(0)

    outputs = Variable(outputs.data, requires_grad=(not eval))
    loss = criterion(outputs.squeeze(-1), targets[0].float().squeeze(-1))

    if not eval:
        # normalize the loss w.r.t batch size.
        loss.div(batch_size).backward()

    gradients = None if outputs.grad is None else outputs.grad.data
    return loss, gradients


def calculate_classification_metrics(
    output: torch.Tensor,
    target: torch.Tensor,
) -> int:
    """Calculate downstream metrics.

    Args:
        output (torch.Tensor): Predicted values generated from the model.
        target (torch.Tensor): Values we want to predict.

    Returns:
        int: Accuracy.
    """
    try:
        outputs = (output > 0.5).transpose(0, 1).long()
        n_correct: int = outputs.eq(target.squeeze(-1)).sum().item()
        return n_correct
    except IndexError as e:
        log.error("error dim", output_shape=output.shape, output=output)
        raise e


def eval(model: ConvNet, criterion: Loss, data: DataLoader):
    total_loss, total_words, total_n_correct = 0, 0, 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(data, desc="Eval"):
            src_seq, _, tgt_seq, _ = map(lambda x: x.to(DEVICE), batch)

            src = src_seq.transpose(0, 1)
            seq_len: int = src.size(0)
            batch_size: int = src.size(1)
            num_words: int = batch_size

            one_hot = Variable(
                torch.FloatTensor(seq_len, batch_size, model.vocab_size).zero_()
            ).to(DEVICE)
            one_hot_scatter = one_hot.scatter_(2, torch.unsqueeze(src, 2), 1)

            outputs = model(one_hot_scatter)
            targets = tgt_seq.transpose(0, 1)
            loss, _ = memory_efficient_loss(outputs, targets, criterion, eval=True)

            total_loss += loss
            total_n_correct += calculate_classification_metrics(outputs, targets)
            total_words += num_words

    return total_loss / total_words, total_n_correct / total_words


def train(
    model: ConvNet, criterion: Loss, data: DataLoader, optim: torch.optim.Optimizer
):
    total_loss, total_words, total_n_correct = 0, 0, 0

    iterator_label = "Train"
    iterator = tqdm(data, desc=iterator_label, leave=True)
    model.train()
    for batch in iterator:
        src_seq, _, tgt_seq, _ = map(lambda x: x.to(DEVICE), batch)

        src = src_seq.transpose(0, 1)
        seq_len: int = src.size(0)
        batch_size: int = src.size(1)
        num_words: int = batch_size  # this is a binary classification task.

        one_hot = Variable(
            torch.FloatTensor(seq_len, batch_size, model.vocab_size).zero_()
        ).to(DEVICE)
        one_hot_scatter = one_hot.scatter_(2, torch.unsqueeze(src, 2), 1)

        model.zero_grad()
        outputs = model(one_hot_scatter)

        targets = tgt_seq.transpose(0, 1)  # shape output to calculate loss.
        loss, gradients = memory_efficient_loss(outputs, targets, criterion)
        outputs.backward(gradients)
        optim.step()  # update the parameters

        n_correct = calculate_classification_metrics(outputs, targets)

        # metrics
        total_loss += loss
        total_n_correct += n_correct
        total_words += num_words
        accuracy = n_correct / num_words

        iterator.set_description(
            f"{iterator_label} Loss={loss.item():.3f}, Accuracy={accuracy:.3f}"
        )
        iterator.refresh()

    return total_loss / total_words, total_n_correct / total_words


def train_model(
    model: ConvNet,
    data_train: DataLoader,
    data_validation: DataLoader,
    dataset: dict,
    optim: torch.optim.Optimizer,
    opt: argparse.Namespace,
    criterion: Loss,
):
    epoch: int
    for epoch in tqdm(range(opt.start_epoch, opt.epochs + 1), desc="Epoch"):
        if opt.extra_shuffle and epoch > opt.curriculum:
            data_train.shuffle()

        train_loss, train_acc = train(model, criterion, data_train, optim)
        log.info("Train Metrics", accuracy=(train_acc * 100), loss=train_loss)

        valid_loss, valid_acc = eval(model, criterion, data_validation)
        log.info("Validation Metrics", accuracy=(valid_acc * 100), loss=valid_loss)

        optim.updateLearningRate(valid_loss, epoch)

        checkpoint_filename: str = "%s_acc_%.2f_loss_%.2f_e%d.pt" % (
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
        opt (Optional[argparse.Namespace], optional): _description_. Defaults to None.
        epoch (Optional[int], optional): _description_. Defaults to 0.
        optim (Optional[torch.optim.Optimizer], optional): _description_. Defaults to None.
        dataset (dict, optional): _description_. Defaults to {}.
    """
    model_state_dict = (
        model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
    )
    model_state_dict = {
        k: v for k, v in model_state_dict.items() if "generator" not in k
    }

    checkpoint = {
        "model": model_state_dict,
        "dicts": dataset["dicts"],
        "opt": opt,
        "epoch": epoch,
        "optim": optim,
    }
    log.info(f"Saving checkpoint to {checkpoint_path}")
    torch.save(
        checkpoint,
        checkpoint_path,
    )
    log.info("Checkpoint successfully saved.")


def main_legacy():
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
    if "kwargs" in dataset["dicts"]["src"]:
        # we're using BPE.
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

    args = namespace_to_dict(opt)
    args["args"] = args  # redundancy option when saving object.
    model: ConvNet = ConvNet(vocab_size=vocabulary_size, **args)

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

    optim.set_parameters(model.parameters())

    if opt.train_from or opt.train_from_state_dict:
        optim.optimizer.load_state_dict(checkpoint["optim"].optimizer.state_dict())

    n_params = sum([p.nelement() for p in model.parameters()])
    log.info("Successfully initialised model weights.", n_params=n_params)

    criterion = nn.BCELoss()

    train_model(model, data_train, data_validation, dataset, optim, opt, criterion)


def init_dataloader(
    dataset: Dataset, tokenizer: Vocabulary, batch_size: int, shuffle=False
) -> DataLoader:
    def tokenize(example):
        # perform tokenization at this stage.
        example["text"] = tokenizer.to_tensor(example["text"])
        return example

    tokenized_dataset = dataset.with_format("torch").map(
        tokenize, desc="Tokenisation", batched=True
    )

    dataloader = DataLoader(
        tokenized_dataset,
        num_workers=8,
        batch_size=batch_size,
        shuffle=shuffle,
        persistent_workers=True,
    )

    return dataloader


def main_new():
    BATCH_SIZE = 128 if DEVICE == "mps" else 680

    # tokenizer
    tokenizer_option = "word"  # "word"
    # interestingly the bpe tokenizer is a lot slwoer to run instead of the word tokenizer.
    # we should see how well this performs with sentencepience.

    if tokenizer_option == "bpe":
        tokenizer_filepath = PROJECT_DIR / "assets" / "tokenizer" / "fr_en_bpe"
        tokenizer = BPEVocabulary.from_file(tokenizer_filepath)
        tokenizer.vocab._progress_bar = iter
    else:
        # word option.
        tokenizer_filepath = PROJECT_DIR / "assets" / "tokenizer" / "fr_en_word_moses"
        tokenizer = WordVocabulary.from_file(tokenizer_filepath)

    log.info("Loaded tokenizer", tokenizer=tokenizer)

    hf_dataset_name = "thien/political"

    dataset: Dataset = load_dataset(hf_dataset_name)
    dataset = dataset.class_encode_column("label")

    train_dataloader = init_dataloader(
        dataset["train"], tokenizer, BATCH_SIZE, shuffle=True
    )
    val_dataloader = init_dataloader(
        dataset["eval"], tokenizer, BATCH_SIZE, shuffle=False
    )

    task = CNNClassifier(
        "political",
        vocabulary_size=tokenizer.vocab_size,
        sequence_length=tokenizer.max_length,
    )

    logger = TensorBoardLogger(LOGGING_DIR, name="pol_classifier")
    logger.log_hyperparams(
        params={
            "batch_size": BATCH_SIZE,
            "tokenizer": tokenizer.to_dict()["class"],
            "dataset": hf_dataset_name,
            "tokenizer_path": tokenizer_filepath,
        }
    )
    
    profiler = AdvancedProfiler(dirpath=logger.log_dir, filename=LOGGING_PERF_NAME)
    trainer = pl.Trainer(
        accelerator=DEVICE,
        max_epochs=1,
        log_every_n_steps=20,
        logger=logger,
        profiler=profiler,
    )

    trainer.fit(
        task, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    # main_legacy()
    main_new()
