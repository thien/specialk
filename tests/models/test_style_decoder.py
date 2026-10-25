import math
import warnings

import pytest
import torch
import torch.nn.functional as F
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from peft import LoraConfig
from peft.utils.peft_types import TaskType
from torch import Tensor
from torch.utils.data import DataLoader

from datasets import Dataset
from specialk.core.constants import PROJECT_DIR, SOURCE, TARGET
from specialk.core.utils import check_torch_device
from specialk.models.classifier.models import BERTClassifier, CNNClassifier
from specialk.models.style_decoder import (
    StyleBackTranslationModelWithBERT,
    StyleBackTranslationModelWithCNN,
)
from specialk.models.tokenizer import SentencePieceVocabulary
from specialk.models.transformer.hf_transformer import MarianMTModule
from specialk.models.transformer.torch.pytorch_transformer import (
    PyTorchTransformerModule,
)

dev_path = "tests/test_files/datasets/en_fr_labels.parquet"
dir_tokenizer = PROJECT_DIR / "assets" / "tokenizer"

torch.manual_seed(1337)


BATCH_SIZE = 50
SEQUENCE_LENGTH = 150


class MetricRecorder(Callback):
    """Capture the metrics Lightning logged, and tau at the end of each batch."""

    def __init__(self):
        super().__init__()
        self.metrics: dict[str, float] = {}
        self.taus: list[float] = []

    def _record(self, trainer: Trainer) -> None:
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, Tensor) and value.numel() == 1:
                self.metrics[key] = float(value.detach().cpu())
            elif isinstance(value, (int, float)):
                self.metrics[key] = float(value)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # on_train_batch_start has already updated tau.
        self.taus.append(float(pl_module.tau))
        self._record(trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._record(trainer)


def make_trainer(**kwargs) -> Trainer:
    options = {
        "max_epochs": 1,
        "logger": False,
        "accelerator": check_torch_device(),
        "limit_train_batches": 1,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "num_sanity_val_steps": 0,
    }
    options.update(kwargs)
    return Trainer(**options)


def make_seq2seq(tokenizer: SentencePieceVocabulary) -> PyTorchTransformerModule:
    return PyTorchTransformerModule(
        name="transformer_1",
        vocabulary_size=tokenizer.vocab_size,
        sequence_length=SEQUENCE_LENGTH,
        dim_model=32,
        n_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        tokenizer=tokenizer,
    )


def make_cnn_classifier(tokenizer: SentencePieceVocabulary) -> CNNClassifier:
    return CNNClassifier(
        name="test_model",
        vocabulary_size=tokenizer.vocab_size,
        sequence_length=SEQUENCE_LENGTH,
    )


@pytest.fixture(scope="session", autouse=True)
def dataset() -> Dataset:
    dataset = Dataset.from_parquet(dev_path)
    return dataset.class_encode_column("label")


@pytest.fixture(scope="session")
def spm_tokenizer() -> SentencePieceVocabulary:
    return SentencePieceVocabulary.from_file(
        dir_tokenizer / "sentencepiece" / "enfr_small.model", max_length=SEQUENCE_LENGTH
    )


@pytest.fixture(scope="session")
def tokenized_dataset(
    dataset: Dataset, spm_tokenizer: SentencePieceVocabulary
) -> Dataset:
    def tokenize(example):
        example["source"] = spm_tokenizer.to_tensor(example["target"])
        example["target"] = spm_tokenizer.to_tensor(example["target"])
        return example

    dataset = dataset.rename_column("text_fr", "source")
    dataset = dataset.rename_column("text", "target")
    return dataset.with_format("torch").map(tokenize)


@pytest.fixture(scope="session")
def train_dataloader(tokenized_dataset: Dataset) -> DataLoader:
    return DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=False)


@pytest.fixture(scope="session")
def val_dataloader(tokenized_dataset: Dataset) -> DataLoader:
    return DataLoader(tokenized_dataset, batch_size=5, shuffle=False)


@pytest.fixture(scope="session")
def spm_dataloader(train_dataloader: DataLoader) -> DataLoader:
    """Alias kept for the Marian trainer-loop tests."""
    return train_dataloader


@pytest.fixture
def seq2seq(spm_tokenizer: SentencePieceVocabulary) -> PyTorchTransformerModule:
    return make_seq2seq(spm_tokenizer)


@pytest.fixture
def cnn_classifier(spm_tokenizer: SentencePieceVocabulary) -> CNNClassifier:
    return make_cnn_classifier(spm_tokenizer)


@pytest.fixture
def cnn_style_module(seq2seq, cnn_classifier) -> StyleBackTranslationModelWithCNN:
    return StyleBackTranslationModelWithCNN(seq2seq, cnn_classifier)


@pytest.fixture(scope="session")
def pretrained_bert() -> BERTClassifier:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BERTClassifier(name="test", model_base_name="bert-base-uncased")


@pytest.fixture
def bert_style_module(seq2seq, pretrained_bert) -> StyleBackTranslationModelWithBERT:
    return StyleBackTranslationModelWithBERT(seq2seq, pretrained_bert)


@pytest.mark.lightweight
def test_cnn_fit_with_validation(cnn_style_module, train_dataloader, val_dataloader):
    recorder = MetricRecorder()
    trainer = make_trainer(callbacks=[recorder], limit_val_batches=1)
    trainer.fit(
        cnn_style_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    metrics = recorder.metrics
    for key in (
        "train_loss",
        "train_recon_loss",
        "train_class_loss",
        "train_acc",
        "tau",
        "eval_loss",
        "eval_recon_loss",
        "eval_class_loss",
        "eval_acc",
        "eval_style_acc",
    ):
        assert key in metrics, f"{key} was not logged (got {sorted(metrics)})"

    assert math.isfinite(metrics["eval_loss"])
    assert metrics["eval_loss"] > 0
    assert 0.0 <= metrics["eval_style_acc"] <= 1.0

    # the logged validation style accuracy must match a manual computation
    # on the same (single) validation batch.
    module = cnn_style_module
    module.eval()
    x, y, _ = module._batch_tensors(next(iter(val_dataloader)))
    x, y = x.to(module.device), y.to(module.device)
    with torch.no_grad():
        seq_logits, _ = module._forward(x, y)
        manual_accuracy = float(module._style_accuracy(seq_logits, y))
    assert metrics["eval_style_acc"] == pytest.approx(manual_accuracy)


@pytest.mark.heavyweight
def test_bert_fit_with_validation(bert_style_module, train_dataloader, val_dataloader):
    recorder = MetricRecorder()
    trainer = make_trainer(callbacks=[recorder], limit_val_batches=1)
    trainer.fit(
        bert_style_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    metrics = recorder.metrics
    for key in (
        "train_loss",
        "train_recon_loss",
        "train_class_loss",
        "train_acc",
        "eval_loss",
        "eval_style_acc",
    ):
        assert key in metrics, f"{key} was not logged (got {sorted(metrics)})"

    assert math.isfinite(metrics["eval_loss"])
    assert 0.0 <= metrics["eval_style_acc"] <= 1.0
    # BERT uses a fixed token map rather than a trainable projection.
    assert bert_style_module.use_mapping is False
    assert bert_style_module.token_map.requires_grad is False
    assert 0.0 <= bert_style_module.mapping_coverage <= 1.0


@pytest.mark.lightweight
def test_frozen_parts_unchanged(cnn_style_module, train_dataloader):
    module = cnn_style_module
    seq2seq = module.sequence_model

    classifier_before = {
        key: value.detach().clone()
        for key, value in module.classifier.state_dict().items()
    }
    encoder_before = {
        key: value.detach().clone()
        for key, value in module._encoder().state_dict().items()
    }
    decoder_before = seq2seq.model.decoder.layers[0].linear1.weight.detach().clone()

    make_trainer().fit(module, train_dataloaders=train_dataloader)

    for key, value in module.classifier.state_dict().items():
        assert torch.equal(value.detach().cpu(), classifier_before[key]), key
    for key, value in module._encoder().state_dict().items():
        assert torch.equal(value.detach().cpu(), encoder_before[key]), key

    decoder_after = seq2seq.model.decoder.layers[0].linear1.weight.detach()
    assert not torch.equal(decoder_after.cpu(), decoder_before), "decoder did not train"

    assert not module.classifier.training
    assert not module._encoder().training
    assert all(not p.requires_grad for p in module.classifier.parameters())
    assert all(not p.requires_grad for p in module._encoder().parameters())


@pytest.mark.lightweight
def test_recon_loss_uses_seq_vocab(seq2seq, cnn_classifier, train_dataloader):
    module = StyleBackTranslationModelWithCNN(
        seq2seq, cnn_classifier, use_mapping=True, mapping_dim=16
    )
    module.eval()

    x, y, _ = module._batch_tensors(next(iter(train_dataloader)))
    with torch.no_grad():
        seq_logits, clf_out = module._forward(x, y)

    assert seq_logits.shape == (x.size(0), y.size(1), seq2seq.vocabulary_size)
    assert module.clf_projection.out_features == cnn_classifier.vocabulary_size
    # the classifier sees one score per example, from its own vocabulary.
    assert clf_out.shape == (x.size(0),)

    recon_loss = module.reconstruction_loss(seq_logits.flatten(0, 1), y.flatten())
    assert torch.isfinite(recon_loss)


@pytest.mark.lightweight
def test_target_label_changes_class_loss(seq2seq, cnn_classifier, train_dataloader):
    module = StyleBackTranslationModelWithCNN(
        seq2seq, cnn_classifier, target_label=0, mapping_dim=16
    )
    module.eval()

    x, y, label = module._batch_tensors(next(iter(train_dataloader)))
    with torch.no_grad():
        _, clf_out = module._forward(x, y)
        loss_label_0 = module.classifier_loss(clf_out, torch.full_like(label, 0))
        loss_label_1 = module.classifier_loss(clf_out, torch.full_like(label, 1))

    assert loss_label_0.ndim == 0 and loss_label_1.ndim == 0
    assert torch.isfinite(loss_label_0) and torch.isfinite(loss_label_1)
    assert not torch.isclose(loss_label_0, loss_label_1)
    assert module.target_label == 0
    assert module.hparams.target_label == 0


@pytest.mark.lightweight
def test_tau_decays(seq2seq, cnn_classifier, tokenized_dataset):
    module = StyleBackTranslationModelWithCNN(
        seq2seq, cnn_classifier, tau_start=2.0, tau_end=0.5
    )
    loader = DataLoader(tokenized_dataset, batch_size=5, shuffle=False)

    recorder = MetricRecorder()
    make_trainer(callbacks=[recorder], limit_train_batches=2).fit(
        module, train_dataloaders=loader
    )

    assert len(recorder.taus) == 2
    assert recorder.taus[0] == pytest.approx(2.0)
    assert recorder.taus[-1] == pytest.approx(0.5, abs=1e-6)
    assert recorder.taus[-1] < recorder.taus[0]
    assert module.tau == pytest.approx(0.5, abs=1e-6)


@pytest.mark.heavyweight
def test_bert_mask(bert_style_module, train_dataloader, monkeypatch):
    module = bert_style_module
    x, y, _ = module._batch_tensors(next(iter(train_dataloader)))

    mask = module._sequence_mask(y)
    assert mask.shape == y.shape
    assert mask.dtype == torch.long
    assert torch.equal(mask.bool(), y.ne(module.seq_pad))
    assert int((y == module.seq_pad).sum()) > 0
    assert mask[y == module.seq_pad].eq(0).all()
    assert mask[y != module.seq_pad].eq(1).all()

    captured: dict[str, Tensor] = {}
    original_forward = module.classifier.model.forward

    def forward(*args, **kwargs):
        captured["attention_mask"] = kwargs.get("attention_mask")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(module.classifier.model, "forward", forward)

    soft = torch.softmax(
        torch.randn(x.size(0), y.size(1), module._classifier_vocabulary_size()),
        dim=-1,
    )
    module.eval()
    with torch.no_grad():
        logits = module._classify(soft, y)

    assert logits.shape == (x.size(0), 2)
    assert captured["attention_mask"] is not None
    assert torch.equal(captured["attention_mask"].bool(), y.ne(module.seq_pad))


@pytest.mark.heavyweight
def test_bert_token_map_is_fixed(bert_style_module, train_dataloader):
    module = bert_style_module
    assert "token_map" not in dict(module.named_parameters())
    assert isinstance(module.token_map, Tensor)
    assert module.token_map.dtype == torch.long

    optimizer = module.configure_optimizers()
    optimizer_ids = {
        id(param) for group in optimizer.param_groups for param in group["params"]
    }
    assert id(module.token_map) not in optimizer_ids
    assert module.seq_projection is None
    assert module.clf_projection is None

    x, y, _ = module._batch_tensors(next(iter(train_dataloader)))
    with torch.no_grad():
        soft = torch.softmax(torch.randn(x.size(0), y.size(1), module.sequence_model.vocabulary_size), dim=-1)
        mapped = module._map(soft)

    assert mapped.shape == (
        x.size(0),
        y.size(1),
        module._classifier_vocabulary_size(),
    )
    assert torch.allclose(
        mapped.sum(-1), torch.ones_like(mapped.sum(-1)), atol=1e-5
    )


@pytest.mark.lightweight
def test_cnn_length_pad_truncate(cnn_style_module, monkeypatch):
    module = cnn_style_module
    module.eval()

    shapes: list[torch.Size] = []
    original_forward = module.classifier.model.forward

    def forward(x):
        shapes.append(x.shape)
        return original_forward(x)

    monkeypatch.setattr(module.classifier.model, "forward", forward)

    batch_size = 4
    vocab_size = module.classifier.vocabulary_size
    for length in (20, SEQUENCE_LENGTH, 200):
        soft = torch.softmax(torch.randn(batch_size, length, vocab_size), dim=-1)
        y = torch.zeros(batch_size, length, dtype=torch.long)
        with torch.no_grad():
            output = module._classify(soft, y)

        assert output.shape == (batch_size,)
        assert torch.isfinite(output).all()
        assert shapes[-1] == (
            module.classifier.sequence_length,
            batch_size,
            vocab_size,
        )


@pytest.mark.lightweight
def test_recon_loss_ignores_seq_pad(cnn_style_module):
    # Marian pads with its last id and uses 0 for </s>, so 0 must still count.
    module = cnn_style_module
    module.seq_pad = 5
    logits = torch.randn(4, 8)
    y = torch.tensor([0, 5, 3, 5])
    expected = F.cross_entropy(logits[[0, 2]], y[[0, 2]])
    assert torch.isclose(module.reconstruction_loss(logits, y), expected)


@pytest.mark.lightweight
def test_cnn_shared_vocab_skips_mapping(cnn_style_module):
    assert cnn_style_module.use_mapping is False
    assert cnn_style_module.seq_projection is None


@pytest.mark.lightweight
def test_optimizer_params(seq2seq, cnn_classifier):
    module = StyleBackTranslationModelWithCNN(seq2seq, cnn_classifier, use_mapping=True)
    optimizer = module.configure_optimizers()

    optimizer_ids = {
        id(param) for group in optimizer.param_groups for param in group["params"]
    }
    trainable_ids = {id(p) for p in module.parameters() if p.requires_grad}

    assert optimizer_ids == trainable_ids
    assert optimizer_ids
    assert all(
        p.requires_grad for group in optimizer.param_groups for p in group["params"]
    )
    assert optimizer.defaults["lr"] == pytest.approx(module.hparams.learning_rate)

    classifier_ids = {id(p) for p in module.classifier.parameters()}
    encoder_ids = {id(p) for p in module._encoder().parameters()}
    assert optimizer_ids.isdisjoint(classifier_ids)
    assert optimizer_ids.isdisjoint(encoder_ids)

    assert id(module.seq_projection.weight) in optimizer_ids
    assert id(module.clf_projection.weight) in optimizer_ids


@pytest.mark.lightweight
def test_transformer_cnn_gradients(spm_tokenizer, train_dataloader):
    """The scene the style decoder relies on: only the decoder gets gradients."""
    seq2seq = make_seq2seq(spm_tokenizer)
    classifier = make_cnn_classifier(spm_tokenizer)

    seq2seq.model.encoder.eval()
    for param in seq2seq.model.encoder.parameters():
        param.requires_grad = False
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False

    batch = next(iter(train_dataloader))
    x = batch[SOURCE].squeeze(1)
    y = batch[TARGET].squeeze(1)
    label = batch["label"]

    tgt_pred = seq2seq.model(x, y)
    y_hat = classifier.model(tgt_pred.transpose(0, 1)).squeeze(-1)

    loss_class = classifier.criterion(y_hat, label.float())
    loss_reconstruction = seq2seq.criterion(
        tgt_pred.reshape(-1, tgt_pred.size(-1)), y.reshape(-1)
    )
    (loss_class + loss_reconstruction).backward()

    encoder_grad = seq2seq.model.encoder.layers[0].linear1.weight.grad
    decoder_grad = seq2seq.model.decoder.layers[0].linear1.weight.grad
    classifier_grad = classifier.model.conv.weight.grad

    assert encoder_grad is None
    assert isinstance(decoder_grad, Tensor)
    assert classifier_grad is None


@pytest.mark.heavyweight
def test_bert_forward_with_class(bert_style_module, train_dataloader):
    module = bert_style_module
    module.eval()

    x, y, label = module._batch_tensors(next(iter(train_dataloader)))
    with torch.no_grad():
        seq_logits, clf_out = module._forward(x, y)

    assert module.use_mapping is False
    assert seq_logits.shape[-1] == module.sequence_model.vocabulary_size
    assert clf_out.shape == (x.size(0), 2)

    recon_loss = module.reconstruction_loss(seq_logits.flatten(0, 1), y.flatten())
    target = torch.full_like(label, module.target_label)
    class_loss = module.classifier_loss(clf_out, target)
    assert torch.isfinite(recon_loss)
    assert torch.isfinite(class_loss)


@pytest.mark.heavyweight
def test_marianmt_bert_fit(spm_dataloader, pretrained_bert):
    seq2seq = MarianMTModule("test_model")
    seq2seq.vocabulary_size = seq2seq.tokenizer.vocab_size

    module = StyleBackTranslationModelWithBERT(seq2seq, pretrained_bert, target_label=1)

    make_trainer().fit(module, train_dataloaders=spm_dataloader)


@pytest.mark.heavyweight
def test_marianmt_lora_bert_fit(spm_dataloader, val_dataloader, pretrained_bert):
    model_base_name = "Helsinki-NLP/opus-mt-fr-en"
    # The helper reads the decoder's layer count off a live module.
    base = MarianMTModule(name=model_base_name, model_base_name=model_base_name)
    modules = base.get_lora_config_modules()
    decoder_layers = len(base.model.model.decoder.layers)
    del base

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=modules,
    )
    seq2seq = MarianMTModule(
        name=model_base_name,
        model_base_name=model_base_name,
        peft_config=peft_config,
    )
    seq2seq.vocabulary_size = seq2seq.tokenizer.vocab_size
    seq2seq._freeze_non_lora_weights()

    # Two weights (A and B) per adapted module.
    lora_names = [
        name for name, _ in seq2seq.model.named_parameters() if "lora_" in name
    ]
    assert len(lora_names) == 2 * len(modules)
    cross_attention = [module for module in modules if "encoder_attn" in module]
    assert len(cross_attention) == 3 * decoder_layers
    for module in cross_attention:
        assert any(module in name for name in lora_names), module

    module = StyleBackTranslationModelWithBERT(seq2seq, pretrained_bert, target_label=1)

    recorder = MetricRecorder()
    trainer = make_trainer(callbacks=[recorder], limit_val_batches=1)
    trainer.fit(
        module,
        train_dataloaders=spm_dataloader,
        val_dataloaders=val_dataloader,
    )

    metrics = recorder.metrics
    assert "eval_loss" in metrics, sorted(metrics)
    assert math.isfinite(metrics["eval_loss"])
    assert 0.0 <= metrics["eval_style_acc"] <= 1.0


@pytest.mark.lightweight
def test_lora_r_requires_marian(tmp_path):
    from specialk.trainers import train_style_decoder

    with pytest.raises(ValueError, match="--lora-r"):
        train_style_decoder.main(
            [
                "--mt-checkpoint",
                str(tmp_path / "mt.ckpt"),
                "--classifier-checkpoint",
                str(tmp_path / "classifier.ckpt"),
                "--lora-r",
                "8",
            ]
        )
