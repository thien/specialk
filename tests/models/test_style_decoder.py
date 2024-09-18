from argparse import Namespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.utils.data import DataLoader

from datasets import Dataset
from specialk import Constants
from specialk.core.utils import log
from specialk.models.classifier.models import BERTClassifier, CNNClassifier
from specialk.models.classifier.onmt.CNNModels import ConvNet
from specialk.models.style_decoder import StyleBackTranslationModel
from specialk.models.tokenizer import SentencePieceVocabulary
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
        example["text"] = spm_tokenizer.to_tensor(example["text"])
        example["text_fr"] = spm_tokenizer.to_tensor(example["text_fr"])
        return example

    tokenized_dataset = dataset.with_format("torch").map(tokenize)

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
    src_text, tgt_text, label = (
        batch["text_fr"].squeeze(1),
        batch["text"].squeeze(1),
        batch["label"],
    )  # not sure why there's additional indexes.

    # sequence 2 sequence forward pass
    tgt_pred_tokens: Float[Tensor, "batch length vocab"] = seq2seq.model(
        src_text, tgt_text
    )

    # if the tokenizers are the same, then you can do this approach.
    classifier_x: Float[Tensor, "seq_len batch d_vocab"]
    classifier_x = tgt_pred_tokens.transpose(0, 1)

    # classifer pass
    y_hat: Float[Tensor, "batch length vocab"] = classifier.model(classifier_x).squeeze(
        -1
    )

    # classifier loss
    loss_class = classifier.criterion(y_hat.squeeze(-1), label.float())

    # reconstruction loss
    loss_reconstruction = seq2seq.criterion(
        tgt_pred_tokens.view(-1, tgt_pred_tokens.size(-1)), tgt_text.view(-1)
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
        batch["text_fr"].squeeze(1),
        batch["text"].squeeze(1),
        batch["label"],
    )  # not sure why there's additional indexes.

    y_hat_text: Float[Tensor, "batch length vocab"]
    y_hat_text = module.nmt_model.model(x, y)

    if module.mapping is not None:
        # tokenizers for the models are different so
        # we need to project that over.
        y_hat_text: Float[Tensor, "length batch vocab_classifier"]
        y_hat_text = y_hat_text.transpose(0, 1) @ module.mapping

    y_hat_label: Float[Tensor, "batch"]
    y_hat_label = module.classifier(y_hat_text)

    # Calculate losses.
    loss_reconstruction = module.reconstruction_loss(y_hat_text, y)
    loss_class = module.classifier_loss(y_hat_label, label)
    joint_loss = loss_class + loss_reconstruction


@pytest.fixture(scope="session", autouse=True)
def pretrained_bert() -> BERTClassifier:
    return BERTClassifier(name="test", model_base_name="bert-base-uncased")


def test_model_inference_transformer_bert_with_class(
    spm_tokenizer, spm_dataloader, pretrained_bert
):
    seq2seq: PyTorchTransformerModule = seq2seq_transformer_model(spm_tokenizer)
    seq2seq.model.encoder.eval()

    # dataset
    # src; tgt; class.
    # pre-filter class.

    classifier: BERTClassifier = pretrained_bert
    classifier.vocabulary_size = classifier.tokenizer.vocab_size
    classifier.refs = Namespace()
    classifier.refs.tgt_label = 1

    module = StyleBackTranslationModel(seq2seq, classifier)

    # classifier stuff
    batch: dict = next(iter(spm_dataloader))
    x, y, label = (
        batch["text_fr"].squeeze(1),
        batch["text"].squeeze(1),
        batch["label"],
    )  # not sure why there's additional indexes.

    y_hat_text: Float[Tensor, "batch length vocab"]
    y_hat_text = module.nmt_model.model(x, y)

    if module.mapping is not None:
        # tokenizers for the models are different so
        # we need to project that over.
        y_hat_embs = (
            y_hat_text
            @ module.mapping
            @ classifier.model.bert.embeddings.word_embeddings.weight
        )

    y_hat_mask = torch.argmax(y_hat_text, dim=-1) == PAD

    y_hat_label = module.classifier.model(
        inputs_embeds=y_hat_embs, attention_mask=y_hat_mask, labels=label
    )

    # Calculate losses.
    loss_reconstruction = module.reconstruction_loss(y_hat_text, y)
    loss_class = y_hat_label.loss
    joint_loss = loss_class + loss_reconstruction
