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
from specialk.models.transformer.pytorch_transformer import PyTorchTransformerModule

dirpath = "tests/tokenizer/test_files"
dev_path = "/Users/t/Projects/datasets/political/political_data/democratic_only.dev.en_fr.parquet"
dir_tokenizer = Constants.PROJECT_DIR / "assets" / "tokenizer"

BATCH_SIZE = 32
SEQUENCE_LENGTH = 100

torch.manual_seed(1337)


@pytest.fixture(scope="session", autouse=True)
def dataset() -> Dataset:
    dataset = Dataset.from_parquet(dev_path)
    dataset = dataset.class_encode_column("label")
    return dataset


@pytest.fixture(scope="session")
def spm_tokenizer() -> SentencePieceVocabulary:
    return SentencePieceVocabulary.from_file(
        dir_tokenizer / "sentencepiece" / "enfr.model", max_length=SEQUENCE_LENGTH
    )


@pytest.fixture(scope="session")
def spm_dataloader(dataset: Dataset, spm_tokenizer: SentencePieceVocabulary):
    def tokenize(example):
        # perform tokenization at this stage.
        example["text"] = spm_tokenizer.to_tensor(example["text"])
        example["text_fr"] = spm_tokenizer.to_tensor(example["text_fr"])
        return example

    tokenized_dataset = dataset.with_format("torch").map(tokenize)
    dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


def seq2seq_model(tokenizer):
    model = PyTorchTransformerModule(
        name="transformer_1",
        vocabulary_size=tokenizer.vocab_size,
        sequence_length=SEQUENCE_LENGTH,
        dim_model=32,
        n_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
    )
    model.PAD = 0

    return model


def test_model_inference(spm_tokenizer, spm_dataloader):
    vocabulary_size = spm_tokenizer.vocab_size
    seq2seq = seq2seq_model(spm_tokenizer)
    # eval doesn't work (i.e. it still stores gradients?)
    seq2seq.model.encoder.eval()

    # disables gradient accumulation @ the encoder.
    for param in seq2seq.model.encoder.parameters():
        param.requires_grad = False

    classifier = ConvNet(
        vocab_size=vocabulary_size,
        sequence_length=SEQUENCE_LENGTH,
    )

    # disables gradient accumulation @ the classifier.
    for param in classifier.parameters():
        param.requires_grad = False

    classifier.eval()
    criterion = nn.BCELoss()

    # classifier stuff
    batch: dict = next(iter(spm_dataloader))
    src_text, tgt_text, label = batch["text_fr"], batch["text"], batch["label"]

    src_text: torch.Tensor = src_text.squeeze(1)
    tgt_text: torch.Tensor = tgt_text.squeeze(1)

    tgt_pred: Float[Tensor, "batch length vocab"]
    tgt_pred = seq2seq.model._forward(src_text, tgt_text)

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

    classifier_x = tgt_pred_tokens.transpose(0, 1)

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

    y_hat = classifier(classifier_x).squeeze(-1)

    # classifier loss
    loss_class = criterion(y_hat.squeeze(-1), label.float())

    # reconstruction loss
    recon_criterion = torch.nn.CrossEntropyLoss(ignore_index=seq2seq.PAD)
    y_hat = tgt_pred_tokens
    loss_reconstruction = recon_criterion(
        y_hat.view(-1, y_hat.size(-1)), tgt_text.view(-1)
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
