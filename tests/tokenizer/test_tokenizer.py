import tempfile
from pathlib import Path

import pytest

from specialk.lib.tokenizer import BPEVocabulary, WordVocabulary

VOCABULARY_SIZE = 1000
SEQUENCE_LENGTH = 100
PCT_BPE = 0.2
SRC_VOCAB = "./tests/test_files/datasets/political_dev.txt"


@pytest.fixture(scope="session", autouse=True)
def word_tokenizer():
    tokenizer = WordVocabulary(
        name="source", vocab_size=VOCABULARY_SIZE, max_length=SEQUENCE_LENGTH
    )
    tokenizer.make(SRC_VOCAB)
    return tokenizer


def test_word_vocabulary(word_tokenizer):
    assert word_tokenizer.tokenize("hello world") == ["hello", "world"]


def test_save_load_word_vocabulary(word_tokenizer):
    dirpath = tempfile.mkdtemp()
    dirpath = "/Users/t/Projects/specialk/tests/tokenizer/test_files"

    tokenizer_filepath = Path(dirpath) / "word_tokenizer"
    word_tokenizer.to_file(tokenizer_filepath)

    new_tokenizer = WordVocabulary.from_file(tokenizer_filepath)
    assert new_tokenizer.vocab.idxToLabel == word_tokenizer.vocab.idxToLabel


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
