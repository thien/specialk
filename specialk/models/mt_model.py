from __future__ import annotations, division

import math
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, TypeVar, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from schedulefree import AdamWScheduleFree
from torch import Tensor

from specialk.core.constants import (
    PAD,
    SOURCE,
    TARGET,
    TEACHER_FORCING_RATIO,
    TEST_ACC,
    TEST_LOSS,
    TRAIN_ACC,
    TRAIN_BATCH_ID,
    TRAIN_LOSS,
    TRAIN_PPLX,
    VAL_ACC,
    VAL_BATCH_ID,
    VAL_BLEU,
    VAL_LOSS,
    VAL_PPLX,
)
from specialk.core.utils import log
from specialk.metrics.metrics import SacreBLEU
from specialk.models.generators.beam import Beam
from specialk.models.ops.ops import mask_out_special_tokens
from specialk.models.recurrent.rnn import Decoder as RNNDecoder
from specialk.models.recurrent.rnn import Encoder as RNNEncoder
from specialk.models.recurrent.rnn import RNNEncoderDecoder as RNNEncoderDecoder
from specialk.models.tokenizer import Vocabulary
from specialk.models.transformer.legacy.Models import Transformer as TransformerModel
from specialk.models.transformer.legacy.Models import (
    TransformerWrapper,
    get_sinusoid_encoding_table,
)
from specialk.models.transformer.legacy.Optim import ScheduledOptim

bleu = SacreBLEU()


class NMTModule(pl.LightningModule):
    """auto-regressive neural machine translation module."""

    def __init__(
        self,
        name: str,
        vocabulary_size: int,
        sequence_length: int,
        label_smoothing: bool = False,
        tokenizer: Optional[Vocabulary] = None,
        decoder_tokenizer: Optional[Vocabulary] = None,
        decoder_vocabulary_size: Optional[int] = None,
        learning_rate: float = 0.001,
        **kwargs,
    ):
        """Initialize the NMT module.

        Args:
            name (str): _description_
            vocabulary_size (int): _description_
            sequence_length (int): _description_
            label_smoothing (bool, optional): _description_. Defaults to False.
            tokenizer (Optional[Vocabulary], optional): _description_. Defaults to None.
            decoder_tokenizer (Optional[Vocabulary], optional):
                Tokenizer used for the decoder only. Defaults to None. If it is set to None,
                then the decoder will use the same vocabulary as the encoder.
            decoder_vocabulary_size (Optional[int], optional): Vocabulary size of the decoder
                generator. Defaults to None. If it is set to none, then the decoder vocabulary
                size is the same as the encoder's vocabulary size.
        """
        super().__init__()
        self.name = name
        self.vocabulary_size = vocabulary_size

        self.sequence_length = sequence_length
        self.label_smoothing = label_smoothing
        self.learning_rate = learning_rate
        self.tokenizer: Union[Vocabulary, None] = tokenizer

        # if we have a separate tokenizer for the decoder,
        # we'll want to make sure that the decoder vocabulary size is set.
        self.decoder_tokenizer: Union[Vocabulary, None] = decoder_tokenizer
        self.decoder_vocabulary_size = decoder_vocabulary_size
        if not self.decoder_tokenizer:
            self.decoder_tokenizer = self.tokenizer
            self.decoder_vocabulary_size = self.vocabulary_size
        else:
            if not self.decoder_vocabulary_size:
                raise Exception(
                    "You have added a decoder tokenizer, "
                    "but you have not set the decoder vocabulary size."
                )

        self.model: Union[TransformerModel, RNNEncoderDecoder]

        # label smoothing isn't needed since the distribution is so wide.
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD)
        self.kwargs = kwargs
        self.save_hyperparameters()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.model(x, y)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Run Training step.

        Args:
            batch (dict): individual batch generated from the DataLoader.
            batch_idx (int): ID corresponding to the batch.

        Returns:
            torch.Tensor: Returns loss.
        """
        x: Int[Tensor, "batch seq_len"] = batch[SOURCE]
        y: Int[Tensor, "batch seq_len"] = batch[TARGET]
        batch_size: int = x.size(0)

        y_hat = self.model(x, y)
        loss = self.loss(y_hat.view(-1, y_hat.size(-1)), y.view(-1))

        n_tokens_correct = self.calculate_classification_metrics(y_hat, y)
        n_tokens_total = y.ne(PAD).sum().item()
        accuracy = n_tokens_correct / n_tokens_total
        perplexity = self.calculate_perplexity(y_hat, y)

        self.log(TRAIN_LOSS, loss, prog_bar=True, batch_size=batch_size)
        self.log_dict(
            {
                TRAIN_ACC: accuracy,
                TRAIN_BATCH_ID: batch_idx,
                TRAIN_PPLX: perplexity,
            },
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
            torch.Tensor: Returns loss.
        """
        x: Int[Tensor, "batch seq_len"] = batch[SOURCE]
        y: Int[Tensor, "batch seq_len"] = batch[TARGET]

        y_hat: Float[Tensor, "batch seq_len vocab"] = self.model(x, y)
        loss = self.loss(y_hat.view(-1, y_hat.size(-1)), y.view(-1))

        n_tokens_correct = self.calculate_classification_metrics(y_hat, y)
        n_tokens_total = y.ne(PAD).sum().item()
        accuracy = n_tokens_correct / n_tokens_total
        perplexity = self.calculate_perplexity(y_hat, y)

        metric_dict = {
            VAL_ACC: accuracy,
            VAL_BATCH_ID: batch_idx,
            VAL_LOSS: loss,
            VAL_PPLX: perplexity,
        }
        if self.decoder_tokenizer is not None:
            metric_dict[VAL_BLEU] = self.validation_bleu(y_hat, y)

        self.log_dict(
            metric_dict,
            batch_size=x.size(0),
        )
        return loss, accuracy

    def validation_bleu(self, y_hat: Tensor, y: Tensor) -> Union[float, None]:
        """Calculates BLEU score @ validation phase.

        Args:
            y_hat (Float[Tensor, "batch seq_len vocab"]): Prediction tensor.
            y (Int[Tensor, "batch seq_len"]): Reference tensor.

        Returns:
            float: BLEU score.
        """
        tokenizer = self.decoder_tokenizer
        if not tokenizer:
            return None

        y_hat = y_hat.argmax(dim=-1)  # greedy as a reference check.
        y_hat = mask_out_special_tokens(y_hat, tokenizer.EOS, tokenizer.PAD)

        predictions: List[str] = tokenizer.detokenize(y_hat, specials=False)
        references: List[str] = tokenizer.detokenize(y, specials=False)
        return bleu.compute(predictions, references)

    def validation_step(self, batch: dict, batch_idx: int):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, batch_size=batch[SOURCE].size(0))
        return metrics

    def test_step(self, batch: dict, batch_idx: int):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {TEST_ACC: acc, TEST_LOSS: loss}
        self.log_dict(metrics, batch_size=batch[SOURCE].size(0))
        return metrics

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if hasattr(self, "lr_scheduler"):
            self.lr_scheduler.step()

    def loss(
        self,
        pred: torch.Tensor,
        ref: torch.Tensor,
        smoothing: bool = False,
        e: float = 0.1,
    ) -> torch.Tensor:
        """Calculate Cross-Entropy Loss.

        Args:
            pred (torch.Tensor): Output generated from model.
            ref (torch.Tensor): Reference values to calculate loss against.
            smoothing (bool, optional): If set, applies smoothing. Defaults to False.
            e (float): epsilon. Defaults to 0.1.

        Returns:
            torch.Tensor: Loss tensor.
        """
        ref = ref.contiguous().view(-1)
        if smoothing:
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, ref.view(-1, 1), 1)
            one_hot = one_hot * (1 - e) + (1 - one_hot) * e / (n_class - 1)

            log_prb = F.log_softmax(pred, dim=1)
            # create non-padding mask with torch.ne()
            non_pad_mask = ref.ne(PAD)
            loss = -(one_hot * log_prb).sum(dim=1)
            # losses are averaged later
            loss = loss.masked_select(non_pad_mask).sum()
        else:
            loss = self.criterion(pred, ref)
            # loss = F.cross_entropy(pred, ref, ignore_index=PAD, reduction="sum")
        return loss

    @staticmethod
    def calculate_classification_metrics(
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> int:
        """Calculate token level accuracy.

        Args:
            output (torch.Tensor): Predicted values generated from the model.
            target (torch.Tensor): Values we want to predict.

        Returns:
            int: Accuracy.
        """
        output = output.argmax(dim=-1)
        non_pad_mask = target.ne(PAD)
        n_correct = output.eq(target)
        n_correct = n_correct.masked_select(non_pad_mask).sum().item()
        return n_correct

    @staticmethod
    def calculate_perplexity(
        y_hat: Float[Tensor, "batch seq vocab"],
        y: Int[Tensor, "batch seq"],
        padding_idx=PAD,
    ) -> float:
        """Calculate perplexity.

        Args:
            y_hat (Tensor): Output log_probs of shape (batch_size, sequence_length, vocab_size).
            y (Tensor): target indices of shape (batch_size, sequence_length).
            padding_idx (int, optional): Index corresponding to pad token. Defaults to PAD.

        Returns:
            float: Perplexity score (scalar).
        """
        # Create a mask to ignore padding tokens
        mask = (y != padding_idx).float()

        # Select log probs of correct tokens
        token_log_probs = y_hat.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)

        # Apply mask and calculate mean negative log likelihood
        masked_log_probs = token_log_probs * mask
        nll = -masked_log_probs.sum() / mask.sum()

        # Calculate perplexity
        return torch.exp(nll).item()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    @torch.inference_mode()
    def generate(
        self,
        input_text: Optional[Union[str, List[str]]],
        output_text: Optional[Union[str, List[str]]],
        input_tokens: Optional[Int[Tensor, "batch seq_len"]],
        output_tokens: Optional[Int[Tensor, "batch seq_len"]],
        **kwargs,
    ) -> Float[torch.Tensor, "batch d_vocab"]:
        """
        Generate next tokens in the sequence, given text and output.

        Args:
            input_text (Union[str, List[str]]): Input token sequence to.

        Note:
            Put either input_text or input_tokens. This is the same as output_text or output_tokens.
        """
        raise NotImplementedError

    def on_after_backward(self):
        """This is called when a backward pass is ran (during training)."""
        # get histogram of parameters.
        if self.global_step % 250 == 0:
            if self.logger is not None:
                # no need to store histograms at every step, that's too expensive.
                for name, param in self.named_parameters():
                    self.logger.experiment.add_histogram(
                        f"{name}_grad", param.grad, self.global_step
                    )
            else:
                log.warn("logger for this object is not initialised through Trainer()")


class TransformerModule(NMTModule):
    def __init__(self, n_warmup_steps: int = 4000, **kwargs):
        super().__init__(**kwargs)
        self.n_warmup_steps = n_warmup_steps
        self.model = TransformerWrapper(
            **kwargs,
        )

    def change_pos_enc_len(self, seq_len: int, pad_token: int = PAD):
        """Change the maximum sequence length of the transformer input.

        Args:
            dimension (int): New sequence length.
        """
        enc_d_word_vec = self.model.encoder.src_word_emb.weight.shape[1]
        dec_d_word_vec = self.model.decoder.tgt_word_emb.weight.shape[1]
        self.model.encoder.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(seq_len, enc_d_word_vec, padding_idx=pad_token),
            freeze=True,
        )
        self.model.decoder.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(seq_len, dec_d_word_vec, padding_idx=pad_token),
            freeze=True,
        )


class RNNModule(NMTModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        args = self.patch_args(**kwargs)
        self.model = RNNEncoderDecoder(
            RNNEncoder(args, self.vocabulary_size),
            RNNDecoder(args, self.decoder_vocabulary_size),
        )
        self.initial_teacher_forcing_ratio: float = 0.9
        self.min_teacher_forcing_ratio: float = 0.5
        self.teacher_forcing_decay_rate: float = (
            3.0  # Adjust this to control decay speed
        )

    @staticmethod
    def patch_args(
        layers: int = 2,
        brnn: bool = False,
        rnn_size: int = 500,
        d_word_vec: int = 300,
        dropout: float = 0.1,
        input_feed: int = 0,
        pre_word_vecs_enc: Optional[Union[Path, str]] = None,
        pre_word_vecs_dec: Optional[Union[Path, str]] = None,
        **kwargs,
    ) -> Namespace:
        """Patch args into Namespace format to feed into model.

        Args:
            layers (int): Number of layers to use.
            brnn (bool): If set, Uses a bidirectional encoder.
            rnn_size (int): Dimension of RNN.
            d_word_vec (int): Dimension size of the token vectors
                representing words (or characters, or bytes).
            dropout (float): Dropout probability' applied between
                self-attention layers/RNN Stacks.
            input_feed (Tensor): Feed the context vector at each time
                step as additional input (via concatenation with the
                word embeddings) to the decoder."
            pre_word_vecs_enc (Union[Path, str]): If a valid path is
                specified, then this will load pretrained word
                embeddings on the encoder side. See README for
                specific formatting instructions.
            pre_word_vecs_dec (Union[Path, str]): If a valid path is
                specified, then this will load pretrained word
                embeddings on the decoder side. See README for
                specific formatting instructions.

        Returns:
            Namespace: Argument namespace containing args in param.
        """
        args = Namespace()
        args.layers = layers
        args.brnn = brnn
        args.rnn_size = rnn_size
        args.d_word_vec = d_word_vec
        args.dropout = dropout
        args.pre_word_vecs_enc = pre_word_vecs_enc
        args.input_feed = input_feed
        args.pre_word_vecs_dec = pre_word_vecs_dec
        return args

    def on_train_epoch_start(self):
        self.update_teacher_forcing_ratio()

    def update_teacher_forcing_ratio(self):
        if self.trainer is not None:
            # Exponential decay formula
            ratio = self.min_teacher_forcing_ratio + (
                self.initial_teacher_forcing_ratio - self.min_teacher_forcing_ratio
            ) * math.exp(
                -self.teacher_forcing_decay_rate
                * self.current_epoch
                / self.trainer.max_epochs
            )

            self.model.decoder.teacher_forcing_ratio = ratio
            # you can observe this in tensorboard.scalars
            self.log(TEACHER_FORCING_RATIO, ratio)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = super().training_step(batch, batch_idx)
        self.log_dict(
            {TEACHER_FORCING_RATIO: self.model.decoder.teacher_forcing_ratio},
            batch_size=batch[SOURCE].size(0),
        )
        return loss

    @torch.inference_mode()
    def generate(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Generate tokens using decoder strategies, such as greedy,
        beam search, top-k, nucleus sampling.
        """
        x = batch[SOURCE]
        batch_size: int = x.shape(0)
        device: str = self.model.device
        max_length: int = self.sequence_length

        # below is a greedy implementation.
        Y = torch.ones(batch_size, 1, device=device)
        probabilities = torch.zeros(batch_size, device=device)  # first token.
        encoder_output = self.model.encoder(x)
        for _ in range(max_length):
            probs_n = self.model.decoder(encoder_output)[:, -1].log_softmax(-1)
            max_probs_n, token_n = probs_n.max(-1)
            token_n = token_n.unsqueeze(-1)
            Y = torch.cat((Y, token_n), axis=1)
            probabilities += max_probs_n
        return Y, probabilities
