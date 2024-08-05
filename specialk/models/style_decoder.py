"""
Style Decoder Model.
"""

from typing import Dict, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader

import specialk.models.classifier.onmt as onmt
import specialk.models.classifier.onmt.CNNModels as CNNModels
from specialk.core.constants import SOURCE, TARGET
from specialk.models.classifier.trainer import CNNClassifier, TextClassifier
from specialk.models.mt_model import NMTModule
from specialk.models.transformer_model import TransformerModel as transformer


class StyleBackTranslationModel(pl.LightningModule):
    def __init__(
        self, mt_model: NMTModule, classifier: CNNClassifier, smoothing: bool = True
    ):
        """
        Args:
            mt_model (NMTModel):
                Machine Translation model (with target language to english).
            cnn_model (CNNModels):
                Style classifier model.
            smoothing (bool, optional):
                If set, adds smothing to reconstruction loss function. Defaults to True.
        """
        self.nmt_model: NMTModule = mt_model
        self.classifier: CNNClassifier = classifier
        self.target_label: int = self.classifier.refs.tgt_label

        # encoder will always be in eval mode. We're only updating the decoder weights.
        self.nmt_model.encoder.eval()
        self.classifier.eval()

        # loss functions.
        self.criterion_classifier = nn.BCELoss()

        self.criterion_class = nn.BCELoss()
        self.criterion_recon = torch.nn.CrossEntropyLoss(
            ignore_index=self.nmt_model.PAD
        )
        self.smoothing = smoothing

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Forward pass of the transformer.
        The output from the model is then sent to the classifier.

        Returns:
            Loss value to run gradients against.
        """

        x: Int[Tensor, "batch seq_len"] = batch[SOURCE]
        y: Int[Tensor, "batch seq_len"] = batch[TARGET]
        label: Int[Tensor, "batch"] = batch["class"]

        # run translation.
        y_hat_text: Float[Tensor, "batch length vocab"] = self.mt_forward(x, y)

        # classifer pass
        y_hat_label: Float[Tensor, "batch"] = self.classifier(y_hat_text).squeeze(-1)

        # classifier loss
        loss_class = self.criterion_class(y_hat_label.squeeze(-1), label.float())

        # reconstruction loss
        loss_reconstruction = self.criterion_recon(
            y_hat_text.view(-1, y_hat_text.size(-1)), y.view(-1)
        )

        # combine losses
        joint_loss = loss_class + loss_reconstruction

        # compute performance
        PAD = self.constants.PAD
        n_tokens_correct = self.n_tokens_correct(y_hat_text, y, pad_token=PAD)
        n_tokens_total = y.ne(PAD).sum().item()
        accuracy = n_tokens_correct / n_tokens_total

        self.log_dict(
            {
                "train_acc": accuracy,
                "batch_id": batch_idx,
                "train_joint_loss": joint_loss,
                "train_recon_loss": loss_reconstruction,
                "train_class_loss": loss_class,
            }
        )
        return joint_loss

    def mt_forward(
        self,
        src_text: Float[Tensor, "batch seq_length"],
        tgt_text: Float[Tensor, "batch seq_length"],
    ) -> Float[Tensor, "batch seq vocab_size"]:
        """
        Forward pass for the Translation Model.
        """
        # sequence 2 sequence forward pass
        tgt_pred: Float[Tensor, "batch seq_length d_model"] = (
            self.nmt_model.model._forward(src_text, tgt_text)
        )
        return F.log_softmax(self.nmt_model.model.generator(tgt_pred), dim=-1)

    def _shared_eval_step(self, batch: DataLoader, batch_idx: int):
        """Evaluation step used for both eval/test runs on a given dataset.

        Args:
            batch (DataLoader): _description_
            batch_idx (int): _description_

        Returns:
            _type_: _description_
        """
        src_seq, src_pos, tgt_seq, tgt_pos = batch

        references = tgt_seq[:, 1:]
        tgt_seq = tgt_seq[:, :-1]
        tgt_pos = tgt_pos[:, :-1]

        # compute encoder output
        encoder_outputs, _ = self.nmt_model.encoder(src_seq, src_pos, True)

        # feed into decoder
        predictions, _, _ = self.nmt_model.decoder(
            tgt_seq, tgt_pos, src_seq, encoder_outputs, True
        )

        # compute performance
        l_recon, l_class, n_correct = self.joint_loss(self, predictions, references)
        l_net = l_recon * l_class

        return l_recon, l_class, l_net, n_correct

    def validation_step(
        self, batch: DataLoader, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the transformer.
        """
        l_recon, l_class, l_net, n_correct = self._shared_eval_step(batch, batch_idx)
        metrics: Dict[str, torch.Tensor] = {
            "val_n_correct": n_correct,
            "val_loss_net": l_net,
            "val_loss_class": l_class,
            "val_loss_reconstruction": l_recon,
        }
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch: DataLoader, batch_idx: int) -> Dict[str, torch.Tensor]:
        l_recon, l_class, l_net, n_correct = self._shared_eval_step(batch, batch_idx)
        metrics: Dict[str, torch.Tensor] = {
            "test_n_correct": n_correct,
            "test_loss_net": l_net,
            "test_loss_class": l_class,
            "test_loss_reconstruction": l_recon,
        }
        self.log_dict(metrics)
        return metrics

    def classifier_loss(self, pred: torch.Tensor):
        """
        Computes classifier loss.
        """

        # need to fix is_cuda
        dim_batch: int
        dim_length: int
        dim_vocab: int

        softmax = nn.Softmax(dim=1)
        seq_length = self.classifier.opt.sequence_length
        target_label = self.classifier.refs.tgt_label

        # translate generator outputs to class inputs
        exp = Variable(pred.data, requires_grad=True, volatile=False)
        linear: torch.FloatTensor = self.class_input(exp)
        dim_batch, dim_length, dim_vocab = linear.size()

        # reshape it for softmax, and shape it back.
        out: torch.FloatTensor = softmax(linear.view(-1, dim_vocab)).view(
            dim_batch, dim_length, dim_vocab
        )

        out = out.transpose(0, 1)
        dim_length, dim_batch, dim_vocab = out.size()

        # setup padding because the CNN only accepts inputs of certain dimension
        if dim_length < seq_length:
            # pad sequences
            cnn_padding = torch.FloatTensor(
                abs(seq_length - dim_length), dim_batch, dim_vocab
            ).zero_()
            cat = torch.cat((out, cnn_padding), dim=0)
        else:
            # trim sequences
            cat = out[:seq_length]

        # get class prediction with cnn
        class_outputs = self.classifier(cat).squeeze()

        # return loss between predicted class outputs and a vector of class_targets
        return self.criterion_classifier(
            class_outputs, torch.FloatTensor(dim_batch).fill_(target_label)
        )

    def joint_loss(
        self, predicted: torch.FloatTensor, reference: torch.FloatTensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Calculates token level accuracy.
        Smoothing can be applied if needed.
        """
        pred = self.model.generator(predicted) * self.model.x_logit_scale

        # compute adversarial loss
        loss_classifier = self.classifier_loss(predicted)

        # compute reconstruction loss
        pred = pred.view(-1, pred.size(2))
        loss_reconstruction = self.nmt_model.calculate_loss(
            pred, reference, self.smoothing
        )

        # count number of correct tokens (for stats)
        n_correct = n_tokens_correct(pred, reference, pad_token=self.constants.PAD)

        return loss_reconstruction, loss_classifier, n_correct

    def configure_optimizers(self):
        # return torch.optim.Adam(self.model.parameters(), lr=0.02)
        return None
