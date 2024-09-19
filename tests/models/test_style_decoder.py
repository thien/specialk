import warnings
from argparse import Namespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichModelSummary
from peft import LoraConfig
from peft.utils.peft_types import TaskType
from torch import Tensor, nn
from torch.utils.data import DataLoader

from datasets import Dataset
from specialk import Constants
from specialk.core.utils import log
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

dirpath = "tests/tokenizer/test_files"
dev_path = "/Users/t/Projects/datasets/political/political_data/democratic_only.dev.en_fr.parquet"
dir_tokenizer = Constants.PROJECT_DIR / "assets" / "tokenizer"

torch.manual_seed(1337)


BATCH_SIZE = 50
SEQUENCE_LENGTH = 150
VOCAB_SIZE_SMALL = 1000
VOCAB_SIZE_BIG = 25000

PAD = 0


@pytest.fixture(scope="session", autouse=True)
def dataset() -> Dataset:
    dataset = Dataset.from_parquet(dev_path)
    dataset = dataset.class_encode_column("label")
    return dataset


@pytest.fixture(scope="session")
def spm_tokenizer() -> SentencePieceVocabulary:
    return SentencePieceVocabulary.from_file(
        dir_tokenizer / "sentencepiece" / "enfr_small.model", max_length=SEQUENCE_LENGTH
    )


@pytest.fixture(scope="session")
def spm_dataloader(dataset: Dataset, spm_tokenizer: SentencePieceVocabulary):
    def tokenize(example):
        # perform tokenization at this stage.
        example["source"] = spm_tokenizer.to_tensor(example["target"])
        example["target"] = spm_tokenizer.to_tensor(example["target"])
        return example

    dataset = dataset.rename_column("text_fr", "source")
    dataset = dataset.rename_column("text", "target")
    tokenized_dataset = dataset.with_format("torch")
    tokenized_dataset = tokenized_dataset.map(tokenize)

    # idk, but the dataset adds an additional dimension
    dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


def seq2seq_transformer_model(tokenizer):
    model = PyTorchTransformerModule(
        name="transformer_1",
        vocabulary_size=tokenizer.vocab_size,
        sequence_length=SEQUENCE_LENGTH,
        dim_model=32,
        n_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
    )
    model.PAD = PAD

    # eval doesn't work (i.e. it still stores gradients?)
    model.model.encoder.eval()

    # disables gradient accumulation @ the encoder.
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    return model


def classifier_small_vocab():
    classifier = CNNClassifier(
        name="test_model",
        vocabulary_size=VOCAB_SIZE_SMALL,
        sequence_length=SEQUENCE_LENGTH,
    )
    classifier.eval()
    # disables gradient accumulation @ the classifier.
    for param in classifier.parameters():
        param.requires_grad = False
    return classifier


def classifier_spm(spm_tokenizer):
    classifier = CNNClassifier(
        name="test_model",
        vocabulary_size=spm_tokenizer.vocab_size,
        sequence_length=SEQUENCE_LENGTH,
    )
    classifier.eval()
    # disables gradient accumulation @ the classifier.
    for param in classifier.parameters():
        param.requires_grad = False
    return classifier


def test_model_inference_transformer(spm_tokenizer, spm_dataloader):
    """
    Test the transformer with a CNN classifier.
    """
    seq2seq = seq2seq_transformer_model(spm_tokenizer)
    classifier = classifier_spm(spm_tokenizer)

    # classifier stuff
    batch: dict = next(iter(spm_dataloader))
    x, y, label = (
        batch["source"].squeeze(1),
        batch["target"].squeeze(1),
        batch["label"],
    )  # not sure why there's additional indexes.

    # sequence 2 sequence forward pass
    tgt_pred_tokens: Float[Tensor, "batch length vocab"] = seq2seq.model(x, y)

    # if the tokenizers are the same, then you can do this approach.
    classifier_x: Float[Tensor, "seq_len batch d_vocab"]
    classifier_x = tgt_pred_tokens.transpose(0, 1)

    # classifer pass
    y_hat: Float[Tensor, "batch length vocab"]
    y_hat = classifier.model(classifier_x).squeeze(-1)

    # classifier loss
    loss_class = classifier.criterion(y_hat.squeeze(-1), label.float())

    # reconstruction loss
    loss_reconstruction = seq2seq.criterion(
        tgt_pred_tokens.view(-1, tgt_pred_tokens.size(-1)), y.view(-1)
    )

    # combine losses
    joint_loss = loss_class + loss_reconstruction
    loss_class.backward()

    encoder_grad = seq2seq.model.encoder.layers[0].linear1.weight.grad
    assert encoder_grad is None
    decoder_grad = seq2seq.model.decoder.layers[0].linear1.weight.grad
    assert isinstance(decoder_grad, Tensor)
    classifier_grad = classifier.model.conv.weight.grad
    assert classifier_grad is None


def test_model_inference_transformer_cnn_with_class(spm_tokenizer, spm_dataloader):
    from argparse import Namespace

    from specialk.models.style_decoder import StyleBackTranslationModel

    seq2seq: PyTorchTransformerModule = seq2seq_transformer_model(spm_tokenizer)
    seq2seq.model.encoder.eval()

    classifier: CNNClassifier = classifier_spm(spm_tokenizer)
    classifier.refs = Namespace()
    classifier.refs.tgt_label = 1

    module = StyleBackTranslationModel(seq2seq, classifier)

    # classifier stuff
    batch: dict = next(iter(spm_dataloader))
    x, y, label = (
        batch["source"].squeeze(1),
        batch["target"].squeeze(1),
        batch["label"],
    )  # not sure why there's additional indexes.

    y_hat_text: Float[Tensor, "batch length vocab"]
    y_hat_text = module.sequence_model.model(x, y)

    if module.mapping is not None:
        # tokenizers for the models are different so
        # we need to project that over.
        y_hat_text: Float[Tensor, "length batch vocab_classifier"]
        y_hat_text = y_hat_text.transpose(0, 1) @ module.mapping

    y_hat_label: Float[Tensor, "batch"]
    y_hat_label = module.classifier(y_hat_text)

    # Calculate losses.
    y_hat_text = y_hat_text.view(-1, y_hat_text.size(-1))
    y = y.view(-1)
    loss_reconstruction = module.reconstruction_loss(y_hat_text, y)

    loss_class = module.classifier_loss(y_hat_label, label)

    joint_loss = loss_class + loss_reconstruction


@pytest.fixture(scope="session", autouse=True)
def pretrained_bert() -> BERTClassifier:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BERTClassifier(name="test", model_base_name="bert-base-uncased")


def test_model_inference_transformer_bert_with_class(
    spm_tokenizer, spm_dataloader, pretrained_bert
):
    seq2seq: PyTorchTransformerModule = seq2seq_transformer_model(spm_tokenizer)
    seq2seq.model.encoder.eval()

    classifier: BERTClassifier = pretrained_bert
    classifier.vocabulary_size = classifier.tokenizer.vocab_size
    classifier.refs = Namespace()
    classifier.refs.tgt_label = 1

    module = StyleBackTranslationModelWithBERT(seq2seq, classifier)

    # classifier stuff
    batch: dict = next(iter(spm_dataloader))
    x, y, label = (
        batch["source"].squeeze(1),
        batch["target"].squeeze(1),
        batch["label"],
    )  # not sure why there's additional indexes.

    y_hat_text, y_hat_label = module._forward(x, y, label)

    # Calculate losses.
    y_hat_text = y_hat_text.view(-1, y_hat_text.size(-1))
    y = y.view(-1)
    loss_reconstruction = module.reconstruction_loss(y_hat_text, y)
    loss_class = y_hat_label.loss

    # this loss class is fundamentally the same.
    assert y_hat_label.loss == module.classifier_loss(y_hat_label.logits, label.long())

    joint_loss = loss_class + loss_reconstruction


def test_model_inference_transformer_cnn_with_class(spm_tokenizer, spm_dataloader):
    seq2seq: PyTorchTransformerModule = seq2seq_transformer_model(spm_tokenizer)
    seq2seq.model.encoder.eval()

    classifier: CNNClassifier = classifier_spm(spm_tokenizer)

    module = StyleBackTranslationModelWithCNN(seq2seq, classifier)

    # classifier stuff
    batch: dict = next(iter(spm_dataloader))
    x, y, label = (
        batch["source"].squeeze(1),
        batch["target"].squeeze(1),
        batch["label"],
    )  # not sure why there's additional indexes.

    y_hat_text, y_hat_label = module._forward(x, y, label)

    # Calculate losses.
    y_hat_text = y_hat_text.view(-1, y_hat_text.size(-1))
    y = y.view(-1)
    loss_reconstruction = module.reconstruction_loss(y_hat_text, y)

    loss_class = module.classifier_loss(y_hat_label, label)

    joint_loss = loss_class + loss_reconstruction


def test_model_inference_transformer_cnn_with_class(spm_tokenizer, spm_dataloader):
    seq2seq: PyTorchTransformerModule = seq2seq_transformer_model(spm_tokenizer)
    seq2seq.model.encoder.eval()

    classifier: CNNClassifier = classifier_spm(spm_tokenizer)

    module = StyleBackTranslationModelWithCNN(seq2seq, classifier)

    # classifier stuff
    batch: dict = next(iter(spm_dataloader))
    x, y, label = (
        batch["source"].squeeze(1),
        batch["target"].squeeze(1),
        batch["label"],
    )  # not sure why there's additional indexes.

    y_hat_text, y_hat_label = module._forward(x, y, label)

    # Calculate losses.
    y_hat_text = y_hat_text.view(-1, y_hat_text.size(-1))
    y = y.view(-1)
    loss_reconstruction = module.reconstruction_loss(y_hat_text, y)

    loss_class = module.classifier_loss(y_hat_label, label)

    joint_loss = loss_class + loss_reconstruction


def test_model_inference_transformer_bert_with_class_trainer_loop(
    spm_tokenizer, spm_dataloader, pretrained_bert
):
    seq2seq: PyTorchTransformerModule = seq2seq_transformer_model(spm_tokenizer)
    seq2seq.model.encoder.eval()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier: BERTClassifier = pretrained_bert
        classifier.vocabulary_size = classifier.tokenizer.vocab_size
        classifier.refs = Namespace()
        classifier.refs.tgt_label = 1

    module = StyleBackTranslationModelWithBERT(seq2seq, classifier)

    trainer = Trainer(
        max_epochs=1, logger=False, limit_train_batches=1, accelerator="cpu"
    )
    trainer.fit(module, train_dataloaders=spm_dataloader)


def test_model_inference_marianmt_bert_with_class_trainer_loop(
    spm_dataloader,
    pretrained_bert,
):
    seq2seq: MarianMTModule = MarianMTModule("test_model")
    seq2seq.model.model.encoder.eval()
    seq2seq.vocabulary_size = seq2seq.tokenizer.vocab_size

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier: BERTClassifier = pretrained_bert
        classifier.vocabulary_size = classifier.tokenizer.vocab_size

    module = StyleBackTranslationModelWithBERT(seq2seq, classifier, target_label=1)

    trainer = Trainer(
        max_epochs=1, logger=False, limit_train_batches=1, accelerator="cpu"
    )
    trainer.fit(module, train_dataloaders=spm_dataloader)


def test_model_inference_lora_marianmt_lora_bert_with_class_trainer_loop(
    spm_dataloader,
    pretrained_bert,
):
    modules = [
        "decoder.layers.0.encoder_attn.q_proj",
        "decoder.layers.0.encoder_attn.k_proj",
        "decoder.layers.0.encoder_attn.v_proj",
        "decoder.layers.1.encoder_attn.q_proj",
        "decoder.layers.1.encoder_attn.k_proj",
        "decoder.layers.1.encoder_attn.v_proj",
        "decoder.layers.2.encoder_attn.q_proj",
        "decoder.layers.2.encoder_attn.k_proj",
        "decoder.layers.2.encoder_attn.v_proj",
        "decoder.layers.3.encoder_attn.q_proj",
        "decoder.layers.3.encoder_attn.k_proj",
        "decoder.layers.3.encoder_attn.v_proj",
        "decoder.layers.4.encoder_attn.q_proj",
        "decoder.layers.4.encoder_attn.k_proj",
        "decoder.layers.4.encoder_attn.v_proj",
        "decoder.layers.5.encoder_attn.q_proj",
        "decoder.layers.5.encoder_attn.k_proj",
        "decoder.layers.5.encoder_attn.v_proj",
    ]
    model_base_name: str = "Helsinki-NLP/opus-mt-fr-en"
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
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
    # freeze all the modules that aren't lota based.
    seq2seq._freeze_non_lora_weights()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier: BERTClassifier = pretrained_bert
        classifier.vocabulary_size = classifier.tokenizer.vocab_size

    module = StyleBackTranslationModelWithBERT(seq2seq, classifier, target_label=1)

    trainer = Trainer(
        max_epochs=1,
        logger=False,
        limit_train_batches=1,
        accelerator="cpu",
        callbacks=[RichModelSummary(max_depth=4)],
    )
    trainer.fit(module, train_dataloaders=spm_dataloader)
