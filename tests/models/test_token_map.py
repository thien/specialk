"""Tests for the fixed seq2seq-to-classifier token map."""

from types import SimpleNamespace

import pytest
import torch

from specialk.models.token_map import build_token_map


class FakeTokenizer:
    """The parts of a HuggingFace tokenizer the map builder reads."""

    def __init__(
        self,
        tokens,
        pad=0,
        unk=1,
        bos=None,
        eos=None,
        cls=None,
        sep=None,
    ):
        self.tokens = tokens
        self.pad_token_id = pad
        self.unk_token_id = unk
        self.bos_token_id = bos
        self.eos_token_id = eos
        self.cls_token_id = cls
        self.sep_token_id = sep

    def convert_ids_to_tokens(self, index):
        return self.tokens[index]


@pytest.mark.lightweight
def test_maps_specials_and_surface_forms():
    seq = FakeTokenizer(
        ["<pad>", "<unk>", "<s>", "</s>", "\u2581hello", "##ing", "\u2581rare"],
        bos=2,
        eos=3,
    )
    clf = FakeTokenizer(
        ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "##ing"],
        cls=2,
        sep=3,
    )

    ids, coverage = build_token_map(seq, clf, 7, 6)

    assert ids.tolist() == [0, 1, 2, 3, 4, 5, 1]
    # Three ordinary seq2seq tokens, two of them matched.
    assert coverage == pytest.approx(2 / 3)


@pytest.mark.lightweight
def test_case_folds_when_the_exact_form_is_missing():
    seq = FakeTokenizer(["<pad>", "<unk>", "\u2581The"])
    clf = FakeTokenizer(["[PAD]", "[UNK]", "the"])

    ids, coverage = build_token_map(seq, clf, 3, 3)

    assert ids.tolist() == [0, 1, 2]
    assert coverage == pytest.approx(1.0)


@pytest.mark.lightweight
def test_unmatched_tokens_point_at_unknown():
    seq = FakeTokenizer(["<pad>", "<unk>", "\u2581zzz"])
    clf = FakeTokenizer(["[PAD]", "[UNK]", "hello"])

    ids, coverage = build_token_map(seq, clf, 3, 3)

    assert ids.tolist() == [0, 1, 1]
    assert coverage == pytest.approx(0.0)


@pytest.mark.lightweight
def test_word_vocabulary_labels_are_read():
    seq = SimpleNamespace(
        vocab=SimpleNamespace(idxToLabel={0: "<blank>", 1: "<unk>", 2: "hello"}),
        PAD=0,
        UNK=1,
    )
    clf = FakeTokenizer(["[PAD]", "[UNK]", "hello"])

    ids, coverage = build_token_map(seq, clf, 3, 3)

    assert ids.tolist() == [0, 1, 2]
    assert coverage == pytest.approx(1.0)


@pytest.mark.lightweight
def test_unknown_tokenizer_raises():
    with pytest.raises(TypeError, match="seq2seq tokenizer"):
        build_token_map(object(), FakeTokenizer(["[PAD]", "[UNK]"]), 2, 2)


@pytest.mark.lightweight
def test_map_is_one_target_per_source_token():
    seq = FakeTokenizer(["<pad>", "<unk>", "\u2581a", "\u2581b"])
    clf = FakeTokenizer(["[PAD]", "[UNK]", "a", "b"])

    ids, _ = build_token_map(seq, clf, 4, 4)

    assert ids.dtype == torch.long
    assert ids.shape == (4,)
    assert int(ids.min()) >= 0
    assert int(ids.max()) < 4
