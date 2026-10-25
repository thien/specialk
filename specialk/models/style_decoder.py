"""
Style Decoder Model.

Fine-tunes a sequence-to-sequence model so that its translations are
classified as the target style. The classifier and the encoder are frozen;
only the decoder is trained. When the classifier uses a different vocabulary,
a fixed token map translates between the two, so the classifier cannot be
satisfied by a map that drifts away from the decoder.
"""

import math

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from peft import PeftModel
from torch import Tensor, nn

from specialk.core.constants import (
    PAD,
    SOURCE,
    TARGET,
    TEST_LOSS,
    TRAIN_LOSS,
    VAL_LOSS,
)
from specialk.models.classifier.models import (
    BERTClassifier,
    CNNClassifier,
    TextClassifier,
)
from specialk.models.mt_model import NMTModule
from specialk.models.ops.ops import token_accuracy
from specialk.models.token_map import build_token_map

# key used for the style accuracy metric at validation time.
STYLE_ACC = "style_acc"


def _pad_or_truncate(x: Tensor, length: int) -> Tensor:
    """Pad (with zeros) or truncate the first dimension of ``x`` to ``length``.

    The CNN's pooling window is fixed to the classifier's sequence length, so
    its input has to have exactly that many time steps.
    """
    current = x.size(0)
    if current == length:
        return x
    if current > length:
        return x[:length]

    padding = x.new_zeros((length - current, *x.shape[1:]))
    return torch.cat([x, padding], dim=0)


class StyleBackTranslationModel(pl.LightningModule):
    def __init__(
        self,
        mt_model: NMTModule,
        classifier: TextClassifier,
        target_label: int = 1,
        use_mapping: bool = True,
        mapping_dim: int = 256,
        lambda_class: float = 1.0,
        tau_start: float = 1.0,
        tau_end: float = 0.1,
        learning_rate: float = 1e-3,
        fixed_token_map: Tensor | None = None,
    ):
        """
        Args:
            mt_model (NMTModule):
                Machine translation model (with target language to english).
            classifier (TextClassifier):
                Style classifier model.
            target_label (int):
                Label the decoder is trained to produce.
            use_mapping (bool):
                Learn a mapping between the sequence model's vocabulary and
                the classifier's vocabulary. Ignored when a fixed map is given.
            mapping_dim (int):
                Dimension of the learned mapping between vocabularies.
            lambda_class (float):
                Weight of the classification loss (thesis lambda_c).
            tau_start (float):
                Initial softmax temperature.
            tau_end (float):
                Final softmax temperature.
            learning_rate (float):
                Learning rate for the trainable parameters.
            fixed_token_map (Tensor | None):
                A ``[seq_vocab]`` table of classifier token ids. When set, the
                mapping is deterministic and never trained.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["mt_model", "classifier", "fixed_token_map"])

        self.sequence_model: NMTModule = mt_model
        self.classifier: TextClassifier = classifier
        self.use_mapping: bool = use_mapping and fixed_token_map is None

        self.target_label: int = target_label
        self.tau: float = tau_start

        # Marian's padding token is not 0, so it can't be a constant.
        self.seq_pad: int = self._sequence_pad()

        self.register_buffer("token_map", fixed_token_map)

        # having one tensor is an unreasonable size; (marianMT is 50k, BERT 30k)...
        self.seq_projection = (
            nn.Linear(self.sequence_model.vocabulary_size, mapping_dim)
            if self.use_mapping
            else None
        )
        self.clf_projection = (
            nn.Linear(mapping_dim, self._classifier_vocabulary_size())
            if self.use_mapping
            else None
        )

        self._freeze_frozen_parts()

    # ------------------------------------------------------------------
    # Frozen parts
    # ------------------------------------------------------------------
    def _sequence_pad(self) -> int:
        """Padding index of the sequence model's tokenizer."""
        tokenizer = getattr(self.sequence_model, "tokenizer", None)
        if tokenizer is None:
            return PAD

        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            return int(pad_token_id)
        return int(getattr(tokenizer, "PAD", PAD))

    def _classifier_vocabulary_size(self) -> int:
        size = self.classifier.vocabulary_size
        if size is not None:
            return int(size)
        # BERT classifiers loaded from a checkpoint don't record it, but the
        # embedding matrix gives the same number.
        return int(self.classifier.model.get_input_embeddings().weight.size(0))

    def _encoder(self) -> nn.Module:
        """The seq2seq encoder.

        ``mt_model.model.encoder`` for our transformer,
        ``mt_model.model.model.encoder`` for Marian.
        """
        model = self.sequence_model.model
        # PEFT's attribute forwarding is unreliable across task types, so
        # unwrap it explicitly.
        if isinstance(model, PeftModel):
            model = model.get_base_model()

        encoder = getattr(model, "encoder", None)
        if encoder is not None:
            return encoder
        return model.model.encoder

    def _uses_peft(self) -> bool:
        return isinstance(self.sequence_model.model, PeftModel)

    def _freeze_frozen_parts(self) -> None:
        for param in self.classifier.parameters():
            param.requires_grad = False

        # PEFT has already frozen the non-LoRA weights; re-freezing would
        # disable the adapters too.
        if not self._uses_peft():
            for param in self._encoder().parameters():
                param.requires_grad = False

        self.classifier.eval()
        self._encoder().eval()

    def train(self, mode: bool = True):
        """Keep the frozen classifier and encoder in eval mode.

        Lightning calls ``.train()`` on the whole module during ``fit``, which
        would otherwise turn the classifier's dropout back on.
        """
        super().train(mode)
        self.classifier.eval()
        self._encoder().eval()
        return self

    # ------------------------------------------------------------------
    # Forward / losses
    # ------------------------------------------------------------------
    def _map(self, soft: Tensor) -> Tensor:
        """Map a distribution over the seq2seq vocabulary into the classifier's."""
        if self.token_map is not None:
            batch, length, _ = soft.shape
            mapped = soft.new_zeros(batch, length, self._classifier_vocabulary_size())
            index = self.token_map.view(1, 1, -1).expand(batch, length, -1)
            return mapped.scatter_add_(2, index, soft)
        if not self.use_mapping:
            return soft
        return F.softmax(self.clf_projection(self.seq_projection(soft)), dim=-1)

    def _forward(
        self,
        x: Int[Tensor, "batch seq_len"],
        y: Int[Tensor, "batch seq_len"],
    ) -> tuple[Float[Tensor, "batch length vocab_seq"], Tensor]:
        """One seq2seq pass, returning seq-vocab logits and the classifier output.

        ``seq_logits`` is the only input to the reconstruction loss; it always
        stays in the seq2seq vocabulary.
        """
        seq_logits: Float[Tensor, "batch length vocab_seq"]
        seq_logits = self.sequence_model(x, y)

        soft: Float[Tensor, "batch length vocab_seq"] = F.softmax(
            seq_logits / self.tau, dim=-1
        )
        clf_out = self._classify(self._map(soft), y)
        return seq_logits, clf_out

    def _classify(
        self,
        soft: Float[Tensor, "batch length vocab_classifier"],
        y: Int[Tensor, "batch seq_len"],
    ) -> Tensor:
        """Run the classifier on the (mapped) distribution. Implemented per subclass."""
        raise NotImplementedError

    def classifier_loss(self, clf_out: Tensor, target: Int[Tensor, " n"]) -> Tensor:
        """Classification loss. Implemented per subclass."""
        raise NotImplementedError

    def reconstruction_loss(self, seq_logits: Tensor, y: Tensor) -> Tensor:
        # NMTModule.criterion ignores index 0, which is Marian's </s>.
        return F.cross_entropy(seq_logits, y, ignore_index=self.seq_pad)

    def _batch_tensors(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        x: Tensor = batch[SOURCE]
        y: Tensor = batch[TARGET]
        label: Tensor = batch["label"]
        # the dataset sometimes adds an extra dimension.
        if x.dim() == 3:
            x = x.squeeze(1)
        if y.dim() == 3:
            y = y.squeeze(1)
        return x, y, label

    @torch.no_grad()
    def _style_accuracy(
        self, seq_logits: Float[Tensor, "batch length vocab_seq"], y: Tensor
    ) -> Tensor:
        """Share of the batch the classifier predicts as ``target_label``."""
        tokens = seq_logits.argmax(dim=-1)
        one_hot = F.one_hot(tokens, num_classes=seq_logits.size(-1)).to(
            seq_logits.dtype
        )
        clf_out = self._classify(self._map(one_hot), y)

        if isinstance(self.classifier, CNNClassifier):
            predictions = (clf_out > 0.5).long()
        else:
            predictions = clf_out.argmax(dim=-1)

        return (predictions == self.target_label).float().mean()

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------
    def _step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        """Shared train/eval step: sequence loss plus classification loss."""
        x, y, label = self._batch_tensors(batch)

        seq_logits, clf_out = self._forward(x, y)

        recon_loss = self.reconstruction_loss(seq_logits.flatten(0, 1), y.flatten())
        target = torch.full_like(label, self.target_label)
        class_loss = self.classifier_loss(clf_out, target)
        loss = recon_loss + self.hparams.lambda_class * class_loss

        self._log_metrics(stage, x, y, seq_logits, loss, recon_loss, class_loss)
        return loss

    def _log_metrics(
        self,
        stage: str,
        x: Tensor,
        y: Tensor,
        seq_logits: Tensor,
        loss: Tensor,
        recon_loss: Tensor,
        class_loss: Tensor,
    ) -> None:
        prefix, loss_name = {
            "train": ("train", TRAIN_LOSS),
            "val": ("eval", VAL_LOSS),
            "test": ("test", TEST_LOSS),
        }[stage]

        metrics: dict[str, Tensor] = {
            loss_name: loss,
            f"{prefix}_recon_loss": recon_loss,
            f"{prefix}_class_loss": class_loss,
            f"{prefix}_acc": token_accuracy(
                seq_logits.flatten(0, 1), y.flatten(), self.seq_pad
            ),
        }

        if stage == "train":
            metrics["tau"] = torch.as_tensor(
                self.tau, dtype=torch.float32, device=x.device
            )
        else:
            metrics[f"{prefix}_{STYLE_ACC}"] = self._style_accuracy(seq_logits, y)

        self.log_dict(
            metrics,
            batch_size=x.size(0),
            on_step=stage == "train",
            on_epoch=True,
        )

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "test")

    def on_train_batch_start(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """Decay tau linearly from ``tau_start`` to ``tau_end`` over training."""
        total_steps = self.trainer.estimated_stepping_batches
        progress = 0.0
        if math.isfinite(total_steps) and total_steps > 1:
            progress = min(self.global_step / (total_steps - 1), 1.0)
        self.tau = (
            self.hparams.tau_start
            + (self.hparams.tau_end - self.hparams.tau_start) * progress
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.hparams.learning_rate)


class StyleBackTranslationModelWithCNN(StyleBackTranslationModel):
    def __init__(
        self,
        mt_model: NMTModule,
        classifier: CNNClassifier,
        **kwargs,
    ):
        assert isinstance(classifier, CNNClassifier), (
            f"Classifier is not a CNNClassifier (found {type(classifier).__name__})"
        )
        # A CNN trained on the seq2seq vocabulary needs no mapping. A CNN with
        # another vocabulary needs a tokenizer to build a fixed map; never a
        # trainable one, which could satisfy the classifier without the decoder.
        seq_tokenizer = mt_model.decoder_tokenizer or mt_model.tokenizer
        if classifier.vocabulary_size != mt_model.vocabulary_size:
            if kwargs.get("use_mapping") is True:
                # An explicit request for the learned map, used by experiments.
                pass
            elif classifier.tokenizer is None or seq_tokenizer is None:
                raise ValueError(
                    "The classifier and seq2seq vocabularies differ, so they need "
                    "a fixed token map; attach the classifier's tokenizer or pass "
                    "a classifier trained on the seq2seq vocabulary"
                )
            else:
                token_map, coverage = build_token_map(
                    seq_tokenizer,
                    classifier.tokenizer,
                    mt_model.vocabulary_size,
                    classifier.vocabulary_size,
                )
                kwargs["fixed_token_map"] = token_map
                kwargs["use_mapping"] = False
                kwargs["mapping_coverage"] = coverage
        else:
            kwargs.setdefault("use_mapping", False)
        coverage = kwargs.pop("mapping_coverage", None)
        super().__init__(mt_model=mt_model, classifier=classifier, **kwargs)
        if coverage is not None:
            self.mapping_coverage = coverage

    def _classify(
        self,
        soft: Float[Tensor, "batch length vocab_classifier"],
        y: Int[Tensor, "batch seq_len"],
    ) -> Float[Tensor, " n"]:
        # ConvNet wants [seq_len, batch, vocab].
        classifier_text = soft.transpose(0, 1)
        classifier_text = _pad_or_truncate(
            classifier_text, self.classifier.sequence_length
        )
        return self.classifier.model(classifier_text).squeeze(-1)

    def classifier_loss(self, clf_out: Tensor, target: Tensor) -> Tensor:
        return self.classifier.criterion(clf_out, target)


class StyleBackTranslationModelWithBERT(StyleBackTranslationModel):
    def __init__(
        self,
        mt_model: NMTModule,
        classifier: BERTClassifier,
        **kwargs,
    ):
        assert isinstance(classifier, BERTClassifier), (
            f"Classifier is not a BERTClassifier (found {type(classifier).__name__})"
        )
        # BERT has its own vocabulary, so the distributions still need
        # translating. The map is fixed: a trained one can raise style accuracy
        # without the decoder improving.
        seq_tokenizer = mt_model.decoder_tokenizer or mt_model.tokenizer
        if seq_tokenizer is None:
            raise ValueError(
                "The BERT style decoder needs the seq2seq tokenizer to build a "
                "fixed token map; load a checkpoint saved with one"
            )
        clf_vocab = classifier.vocabulary_size or int(
            classifier.model.get_input_embeddings().weight.size(0)
        )
        token_map, coverage = build_token_map(
            seq_tokenizer,
            classifier.tokenizer,
            mt_model.vocabulary_size,
            clf_vocab,
        )
        kwargs["use_mapping"] = False
        kwargs["fixed_token_map"] = token_map
        super().__init__(mt_model=mt_model, classifier=classifier, **kwargs)
        # Share of ordinary seq2seq tokens that found a BERT token by surface
        # form. The rest become [UNK]; log it so a run can be judged with it.
        self.mapping_coverage = coverage

    def _sequence_mask(
        self, y: Int[Tensor, "batch seq_len"]
    ) -> Int[Tensor, "batch seq_len"]:
        """1 on non-padding tokens, 0 on padding."""
        return (y != self.seq_pad).long()

    def _classify(
        self,
        soft: Float[Tensor, "batch length vocab_classifier"],
        y: Int[Tensor, "batch seq_len"],
    ) -> Float[Tensor, "batch 2"]:
        # soft is a distribution over the classifier's vocabulary, so its
        # matrix product with the embedding table is the expected embedding.
        embeddings = soft @ self.classifier.model.get_input_embeddings().weight
        return self.classifier.model(
            inputs_embeds=embeddings, attention_mask=self._sequence_mask(y)
        ).logits

    def classifier_loss(self, clf_out: Tensor, target: Tensor) -> Tensor:
        return self.classifier.criterion(clf_out, target.long())
