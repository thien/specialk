import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from specialk.core.constants import PROJECT_DIR
from specialk.core.utils import log
from specialk.datasets.preprocess import load_file
from specialk.models.mt_model import NMTModule
from specialk.models.tokenizer import (
    BPEVocabulary,
    SentencePieceVocabulary,
    WordVocabulary,
)

VOCABULARY_SIZE = 1000
SEQUENCE_LENGTH = 100
PCT_BPE = 0.2
SRC_VOCAB = "./tests/test_files/datasets/political_dev.txt"


@pytest.fixture(scope="session", autouse=True)
def word_tokenizer():
    tokenizer = WordVocabulary(
        name="source", vocab_size=VOCABULARY_SIZE, max_length=SEQUENCE_LENGTH
    )
    text = load_file(SRC_VOCAB, None)
    tokenizer.fit(text)
    return tokenizer


def test_word_tokenizer_seq_len_change(word_tokenizer):
    text = ["hello world"]
    tensor_before = word_tokenizer.to_tensor(text)
    new_size = 28
    log.info("i'm here")
    word_tokenizer.max_length = new_size
    tensor_after = word_tokenizer.to_tensor(text)
    assert tensor_after.shape == torch.Size([1, new_size])
    assert tensor_after.shape != tensor_before.shape


def test_word_vocabulary(word_tokenizer):
    assert word_tokenizer.tokenize("hello world") == ["hello", "world"]


def test_save_load_word_vocabulary(word_tokenizer):
    dirpath = tempfile.mkdtemp()
    dirpath = "/Users/t/Projects/specialk/tests/tokenizer/test_files"

    tokenizer_filepath = Path(dirpath) / "word_tokenizer"
    word_tokenizer.to_file(tokenizer_filepath)

    new_tokenizer = WordVocabulary.from_file(tokenizer_filepath)
    assert new_tokenizer.vocab.idxToLabel == word_tokenizer.vocab.idxToLabel


def test_word_tokenizer_to_tensor(word_tokenizer):
    sequences = ["hello world", "hello"]
    output = word_tokenizer.to_tensor(sequences)
    max_len = word_tokenizer.max_length
    assert output.shape[-1] == max_len


@pytest.fixture(scope="session", autouse=True)
def pol_word_tokenizer():
    tokenizer_filepath = PROJECT_DIR / "assets" / "tokenizer" / "fr_en_word_moses"
    return WordVocabulary.from_file(tokenizer_filepath)


def test_word_tokenizer_to_tensor_long(pol_word_tokenizer):
    sequence = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    output = pol_word_tokenizer.to_tensor(sequence)
    max_len = pol_word_tokenizer.max_length
    print(pol_word_tokenizer, max_len)
    print(output.shape)
    assert output.shape[-1] == max_len


@pytest.fixture(scope="session", autouse=True)
def bpe_tokenizer():
    tokenizer = BPEVocabulary(
        name="source",
        vocab_size=VOCABULARY_SIZE,
        max_length=SEQUENCE_LENGTH,
        pct_bpe=PCT_BPE,
    )
    tokenizer.make(SRC_VOCAB)
    return tokenizer


def test_bpe_vocabulary(bpe_tokenizer):
    print(bpe_tokenizer.tokenize("hello world"))
    assert bpe_tokenizer.tokenize("hello world") == [
        "__sow",
        "he",
        "ll",
        "o",
        "__eow",
        "world",
    ]


def test_save_load_bpe_vocabulary(bpe_tokenizer):
    dirpath = tempfile.mkdtemp()
    dirpath = "/Users/t/Projects/specialk/tests/tokenizer/test_files"

    tokenizer_filepath = Path(dirpath) / "bpe_tokenizer"
    bpe_tokenizer.to_file(tokenizer_filepath)

    new_tokenizer = BPEVocabulary.from_file(tokenizer_filepath)
    assert new_tokenizer.vocab.bpe_vocab == bpe_tokenizer.vocab.bpe_vocab
    assert new_tokenizer.vocab.word_vocab == bpe_tokenizer.vocab.word_vocab
    assert new_tokenizer.vocab.required_tokens == bpe_tokenizer.vocab.required_tokens
    assert new_tokenizer.vocab.strict == bpe_tokenizer.vocab.strict


@pytest.fixture(scope="session", autouse=True)
def sentencepiece_tokenizer():
    spm_path = str(
        PROJECT_DIR / "assets" / "tokenizer" / "sentencepiece" / "enfr.model"
    )
    tokenizer = SentencePieceVocabulary.from_file(spm_path, max_length=100)
    return tokenizer


def test_spm_tokenizer_to_tensor_long(sentencepiece_tokenizer):
    sequence = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    output = sentencepiece_tokenizer.to_tensor(sequence)
    max_len = sentencepiece_tokenizer.max_length
    print(sentencepiece_tokenizer, max_len)
    print(output.shape)
    assert output.shape[-1] == max_len


def test_spm_tokenizer_encode_decode(sentencepiece_tokenizer):
    sequence = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam"
    tokens = sentencepiece_tokenizer.to_tensor(sequence)
    assert tokens != sequence
    _sequence = sentencepiece_tokenizer.detokenize(tokens)
    print(_sequence)
    assert _sequence == sequence


def test_model_validation_bleu_spm(sentencepiece_tokenizer):
    src_tokenizer = sentencepiece_tokenizer
    model = NMTModule(
        name="test",
        vocabulary_size=src_tokenizer.vocab_size,
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=src_tokenizer,
    )

    _refs = ["The dog bit the man.", "It was not unexpected.", "The man bit him first."]

    _hyps = [
        "The dog bit the man.",
        "It wasn't surprising.",
        "The man had just bitten him.",
    ]

    ref_tensor = [src_tokenizer.to_tensor(i) for i in _refs]
    hyp_tensor = [src_tokenizer.to_tensor(i) for i in _hyps]

    ref_tensor = torch.cat(ref_tensor)
    hyp_tensor = torch.cat(hyp_tensor)

    batch_size, seq_len = ref_tensor.shape

    # with the ref_tensor, turn that into one hot and then fill
    # it in with the token indicated (this is approximate the model generation)
    hyp_one_hot = torch.zeros(
        batch_size,
        seq_len,
        src_tokenizer.vocab_size,
    ).scatter_(2, torch.unsqueeze(hyp_tensor, 2), 1)

    assert (hyp_tensor.shape[0], hyp_tensor.shape[1]) == (batch_size, seq_len)
    assert torch.all(hyp_one_hot.argmax(dim=-1).eq(hyp_tensor))

    pred_score = model.validation_bleu(hyp_one_hot, ref_tensor)

    score = 45.06

    assert np.isclose(pred_score, score, rtol=1e-02, atol=1e-03)


def test_model_validation_bleu_word(word_tokenizer):
    src_tokenizer = word_tokenizer
    model = NMTModule(
        name="test",
        vocabulary_size=src_tokenizer.vocab_size,
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=src_tokenizer,
    )

    _refs = ["The dog bit the man.", "It was not unexpected.", "The man bit him first."]

    _hyps = [
        "The dog bit the man.",
        "It wasn't surprising.",
        "The man had just bitten him.",
    ]

    ref_tensor = [src_tokenizer.to_tensor(i) for i in _refs]
    hyp_tensor = [src_tokenizer.to_tensor(i) for i in _hyps]

    ref_tensor = torch.cat(ref_tensor).view(len(_refs), -1)
    hyp_tensor = torch.cat(hyp_tensor).view(len(_hyps), -1)

    batch_size, seq_len = ref_tensor.shape

    # with the ref_tensor, turn that into one hot and then fill
    # it in with the token indicated (this is approximate the model generation)
    hyp_one_hot = torch.zeros(
        batch_size,
        seq_len,
        src_tokenizer.vocab_size,
    ).scatter_(2, torch.unsqueeze(hyp_tensor, 2), 1)

    assert (hyp_tensor.shape[0], hyp_tensor.shape[1]) == (batch_size, seq_len)
    assert torch.all(hyp_one_hot.argmax(dim=-1).eq(hyp_tensor))

    pred_score = model.validation_bleu(hyp_one_hot, ref_tensor)

    score = 90

    assert np.isclose(pred_score, score, rtol=5, atol=5)
