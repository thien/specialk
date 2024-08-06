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
from specialk.models.classifier.onmt.CNNModels import ConvNet
from specialk.models.tokenizer import SentencePieceVocabulary
from specialk.models.transformer.torch.pytorch_transformer import (
    PyTorchTransformerModule,
)

dirpath = "tests/tokenizer/test_files"
dev_path = "/Users/t/Projects/datasets/political/political_data/democratic_only.dev.en_fr.parquet"
dir_tokenizer = Constants.PROJECT_DIR / "assets" / "tokenizer"

BATCH_SIZE = 32
SEQUENCE_LENGTH = 100

torch.manual_seed(1337)

VOCAB_SIZE_SMALL = 1000
VOCAB_SIZE_BIG = 5000

PAD = 0

criterion_class = nn.BCELoss()
crirerion_recon = torch.nn.CrossEntropyLoss(ignore_index=PAD)


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
    classifier = ConvNet(
        vocab_size=VOCAB_SIZE_SMALL,
        sequence_length=SEQUENCE_LENGTH,
    )
    classifier.eval()
    # disables gradient accumulation @ the classifier.
    for param in classifier.parameters():
        param.requires_grad = False
    return classifier


def classifier_spm(spm_tokenizer):
    classifier = ConvNet(
        vocab_size=spm_tokenizer.vocab_size,
        sequence_length=SEQUENCE_LENGTH,
    )
    classifier.eval()
    # disables gradient accumulation @ the classifier.
    for param in classifier.parameters():
        param.requires_grad = False
    return classifier


def test_model_inference_transformer(spm_tokenizer, spm_dataloader):
    vocabulary_size = spm_tokenizer.vocab_size
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
    tgt_pred: Float[Tensor, "batch length vocab"] = seq2seq.model._forward(
        src_text, tgt_text
    )
    tgt_pred_tokens: Float[Tensor, "batch seq generator"] = F.log_softmax(
        seq2seq.model.generator(tgt_pred), dim=-1
    )

    # the forward pass

    # wrap into one-hot encoding of tokens for activation.
    # this implementation won't pass gradients because of the argmax.
    # x_argmax = torch.argmax(tgt_pred, dim=-1)
    # classifier_x: Float[Tensor, "length batch vocab"]
    # classifier_x = torch.zeros(SEQUENCE_LENGTH, BATCH_SIZE, vocabulary_size).scatter_(
    #    2, torch.unsqueeze(x_argmax.T, 2), 1
    # )

    # if the tokenizers are the same, then you can do this approach.
    classifier_x = tgt_pred_tokens.transpose(0, 1)

    # otherwise you need to learn a mapping on top.
    # also throw a warning out.

    """
    method here is the original(ish) implementation
    """
    # TODO: you need to learn a different projection from the decoder to the tokenizer's input space.
    # class_input = nn.Linear(
    #     seq2seq.model.generator.weight.shape[1],
    #     classifier.word_lut.weight.shape[0],
    # )

    # linear: Tensor = class_input(tgt_pred)
    # dim_batch, dim_length, dim_vocab = linear.size()

    # # reshape it for softmax, and shape it back.
    # out: Tensor = F.softmax(linear.view(-1, dim_vocab), dim=-1).view(
    #     dim_batch, dim_length, dim_vocab
    # )

    # out = out.transpose(0, 1)
    # dim_length, dim_batch, dim_vocab = out.size()

    # # setup padding because the CNN only accepts inputs of certain dimension
    # if dim_length < SEQUENCE_LENGTH:
    #     # pad sequences
    #     cnn_padding = torch.zeros(
    #         (abs(SEQUENCE_LENGTH - dim_length), dim_batch, dim_vocab)
    #     )
    #     cat = torch.cat((out, cnn_padding), dim=0)
    # else:
    #     # trim sequences
    #     cat = out[:SEQUENCE_LENGTH]
    # classifier_x = cat

    # classifer pass
    y_hat: Float[Tensor, "batch length vocab"] = classifier(classifier_x).squeeze(-1)

    # classifier loss
    loss_class = criterion_class(y_hat.squeeze(-1), label.float())

    # reconstruction loss
    loss_reconstruction = crirerion_recon(
        tgt_pred_tokens.view(-1, tgt_pred_tokens.size(-1)), tgt_text.view(-1)
    )

    # combine losses
    joint_loss = loss_class + loss_reconstruction
    loss_class.backward()

    encoder_grad = seq2seq.model.encoder.layers[0].linear1.weight.grad
    assert encoder_grad is None
    decoder_grad = seq2seq.model.decoder.layers[0].linear1.weight.grad
    assert isinstance(decoder_grad, Tensor)
    classifier_grad = classifier.conv.weight.grad
    assert classifier_grad is None
