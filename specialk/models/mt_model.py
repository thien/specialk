from __future__ import division

from typing import Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from specialk.models.tokenizer import Vocabulary
from torch import Tensor
from specialk.metrics.metrics import SacreBLEU
from specialk.models.transformer.Optim import ScheduledOptim

from specialk.models.transformer.Models import Transformer as TransformerModel
from specialk.models.transformer.Models import get_sinusoid_encoding_table

from specialk.models.recurrent.Models import NMTModel as Seq2Seq, Encoder, Decoder

from specialk.core.constants import PAD

bleu = SacreBLEU()


class NMTModule(pl.LightningModule):
    """auto-regressive neural machine translation module."""

    def __init__(
        self,
        name: str,
        vocabulary_size: int,
        sequence_length: int,
        label_smoothing: bool = False,
        tokenizer: Vocabulary | None = None,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.vocabulary_size = vocabulary_size
        self.sequence_length = sequence_length
        self.label_smoothing = label_smoothing
        self.tokenizer: Vocabulary | None = tokenizer

        self.model: Union[TransformerModel, Seq2Seq]

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Run Training step.

        Args:
            batch (dict): individual batch generated from the DataLoader.
            batch_idx (int): ID corresponding to the batch.

        Returns:
            torch.Tensor: Returns loss.
        """
        x, y = batch["source"], batch["target"]
        batch_size: int = x.size(0)

        y_hat = self.model(x).squeeze(-1)
        loss = self.criterion(y_hat, y)

        n_tokens_correct = self.calculate_classification_metrics(y_hat, y)
        n_tokens_total = y.ne(self.constants.PAD).sum().item()
        accuracy = n_tokens_correct / n_tokens_total

        self.log_dict(
            {
                "train_acc": accuracy,
                "batch_id": batch_idx,
                "train_loss": loss,
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
        x: Int[Tensor, "batch seq_len"] = batch["source"]
        y: Int[Tensor, "batch seq_len"] = batch["target"]

        y_hat: Float[Tensor, "batch seq_len vocab"] = self.model(x)
        loss = self.criterion(y_hat, y)

        n_tokens_correct = self.calculate_classification_metrics(y_hat, y)
        n_tokens_total = y.ne(self.constants.PAD).sum().item()
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
        predictions = self.tokenizer.detokenize(y_hat)
        references = self.tokenizer.detokenize(y)
        return bleu.compute(predictions, references)

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
            non_pad_mask = ref.ne(self.constants.PAD)
            loss = -(one_hot * log_prb).sum(dim=1)
            # losses are averaged later
            loss = loss.masked_select(non_pad_mask).sum()
        else:
            loss = F.cross_entropy(
                pred, ref, ignore_index=self.constants.PAD, reduction="sum"
            )
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
        output = output.max(1)[1]
        target = target.contiguous().view(-1)
        non_pad_mask = target.ne(self.constants.PAD)
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = Encoder(vocabulary_size=self.vocabulary_size)
        self.decoder = Decoder(vocabulary_size=self.vocabulary_size)
        self.model = Seq2Seq(self.encoder, self.decoder)
