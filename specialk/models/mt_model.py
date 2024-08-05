from __future__ import division

from argparse import Namespace
from pathlib import Path
from typing import Optional, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from specialk.core.constants import PAD
from specialk.metrics.metrics import SacreBLEU
from specialk.models.ops import mask_out_special_tokens
from specialk.models.recurrent.Models import Decoder as RNNDecoder
from specialk.models.recurrent.Models import Encoder as RNNEncoder
from specialk.models.recurrent.Models import NMTModel as Seq2Seq
from specialk.models.tokenizer import Vocabulary
from specialk.models.transformer.legacy.Models import Transformer as TransformerModel
from specialk.models.transformer.legacy.Models import get_sinusoid_encoding_table
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
        tokenizer: Union[Vocabulary, None] = None,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.vocabulary_size = vocabulary_size
        self.sequence_length = sequence_length
        self.label_smoothing = label_smoothing
        self.tokenizer: Union[Vocabulary, None] = tokenizer

        self.model: Union[TransformerModel, Seq2Seq]
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Run Training step.

        Args:
            batch (dict): individual batch generated from the DataLoader.
            batch_idx (int): ID corresponding to the batch.

        Returns:
            torch.Tensor: Returns loss.
        """
        x: Int[Tensor, "batch seq_len"] = batch["source"]
        y: Int[Tensor, "batch seq_len"] = batch["target"]

        y_hat = self.model(x, y)
        loss = self.loss(y_hat.view(-1, y_hat.size(-1)), y.view(-1))

        n_tokens_correct = self.calculate_classification_metrics(y_hat, y)
        n_tokens_total = y.ne(PAD).sum().item()
        accuracy = n_tokens_correct / n_tokens_total

        self.log_dict(
            {
                "train_acc": accuracy,
                "batch_id": batch_idx,
                "train_loss": loss,
            },
            batch_size=x.size(0),
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
        x: Int[Tensor, "batch seq_len"] = batch["source"]
        y: Int[Tensor, "batch seq_len"] = batch["target"]

        y_hat: Float[Tensor, "batch seq_len vocab"] = self.model(x, y)
        loss = self.loss(y_hat.view(-1, y_hat.size(-1)), y.view(-1))

        n_tokens_correct = self.calculate_classification_metrics(y_hat, y)
        n_tokens_total = y.ne(PAD).sum().item()
        accuracy = n_tokens_correct / n_tokens_total

        metric_dict = {
            "eval_acc": accuracy,
            "batch_id": batch_idx,
            "eval_loss": loss,
        }
        if self.tokenizer is not None:
            metric_dict["bleu"] = self.validation_bleu(y_hat, y)

        self.log_dict(
            metric_dict,
            batch_size=x.size(0),
        )
        return loss, accuracy

    def validation_bleu(self, y_hat: Tensor, y: Tensor) -> float:
        """Calculates BLEU score @ validation phase.

        Args:
            y_hat (Int[Tensor, "batch seq_len vocab"]): Prediction tensor.
            y (Int[Tensor, "batch seq_len"]): Reference tensor.

        Returns:
            float: BLEU score.
        """
        if not self.tokenizer:
            return None
        y_hat = y_hat.argmax(dim=-1)
        y_hat = mask_out_special_tokens(y_hat, self.tokenizer.EOS, self.tokenizer.PAD)
        predictions = self.tokenizer.detokenize(y_hat.tolist())
        references = self.tokenizer.detokenize(y.tolist())
        return bleu.compute(predictions, references)

    def validation_step(self, batch: dict, batch_idx: int):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, batch_size=batch["source"].size(0))
        return metrics

    def test_step(self, batch: dict, batch_idx: int):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, batch_size=batch["source"].size(0))
        return metrics

    def predict_step(
        self, batch: dict, batch_idx: int, dataloader_idx=0
    ) -> torch.Tensor:
        raise NotImplementedError

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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=0.02)

    def generate(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Generate tokens using decoder strategies, such as greedy,
        beam search, top-k, nucleus sampling.
        """
        x = batch["source"]
        batch_size: int = x.shape(0)
        device: str = self.model.device
        max_length: int = self.sequence_length

        # below is a greedy implementation.
        with torch.no_grad():
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


class TransformerModule(NMTModule):
    def __init__(self, n_warmup_steps: int = 4000, **kwargs):
        super().__init__(**kwargs)
        self.n_warmup_steps = n_warmup_steps
        self.model = TransformerModel(
            n_src_vocab=self.vocabulary_size,
            n_tgt_vocab=self.vocabulary_size,
            len_max_seq=self.sequence_length,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return ScheduledOptim(
            optimizer=torch.optim.Adam(
                filter(lambda x: x.requires_grad, self.model.parameters()),
                betas=(0.9, 0.98),
                eps=1e-09,
                lr=0.01,
            ),
            d_model=self.model.encoder.d_model,
            n_warmup_steps=self.n_warmup_steps,
        )

    def change_sequence_length(self, size: int, pad_token: int = PAD):
        """Change the maximum sequence length of the transformer input.

        Args:
            dimension (int): New sequence length.
        """
        enc_d_word_vec = self.model.encoder.src_word_emb.weight.shape[1]
        dec_d_word_vec = self.model.decoder.tgt_word_emb.weight.shape[1]
        self.model.encoder.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(size, enc_d_word_vec, padding_idx=pad_token),
            freeze=True,
        )
        self.model.decoder.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(size, dec_d_word_vec, padding_idx=pad_token),
            freeze=True,
        )


class RNNModule(NMTModule):
    def __init__(self, vocabulary_size: int, **kwargs):
        super().__init__(vocabulary_size=vocabulary_size, **kwargs)
        args = self.patch_args(**kwargs)
        self.model = Seq2Seq(
            RNNEncoder(args, vocabulary_size),
            RNNDecoder(args, vocabulary_size),
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
