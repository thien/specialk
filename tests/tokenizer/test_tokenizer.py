from specialk.lib.tokenizer import WordVocabulary, BPEVocabulary
from pathlib import Path
import tempfile
import pytest

VOCABULARY_SIZE = 1000
SEQUENCE_LENGTH = 100
PCT_BPE = 0.2
SRC_VOCAB = "./tests/test_files/datasets/political_dev.txt"


@pytest.fixture(scope="session", autouse=True)
def word_tokenizer():
    tokenizer = WordVocabulary("source", "", VOCABULARY_SIZE, SEQUENCE_LENGTH, PCT_BPE)
    tokenizer.make(SRC_VOCAB)
    return tokenizer


def test_word_vocabulary(word_tokenizer):
    assert word_tokenizer.tokenize("hello world") == ["hello", "world"]


def test_save_load_word_vocabulary(word_tokenizer):
    dirpath = tempfile.mkdtemp()
    dirpath = "/Users/t/Projects/specialk/tests/tokenizer/test_files"

    tokenizer_filepath = Path(dirpath) / "word_tokenizer"
    word_tokenizer.save(tokenizer_filepath)

    new_tokenizer = WordVocabulary(
        "source", tokenizer_filepath, VOCABULARY_SIZE, SEQUENCE_LENGTH, PCT_BPE
    )
    new_tokenizer.load()
    assert new_tokenizer.vocab.idxToLabel == word_tokenizer.vocab.idxToLabel


@pytest.fixture(scope="session", autouse=True)
def bpe_tokenizer():
    tokenizer = BPEVocabulary("source", "", VOCABULARY_SIZE, SEQUENCE_LENGTH, PCT_BPE)
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
    bpe_tokenizer.save(tokenizer_filepath)

    new_tokenizer = BPEVocabulary(
        "source", tokenizer_filepath, VOCABULARY_SIZE, SEQUENCE_LENGTH, PCT_BPE
    )
    new_tokenizer.load()
    assert new_tokenizer.vocab.bpe_vocab == bpe_tokenizer.vocab.bpe_vocab