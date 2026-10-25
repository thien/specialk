"""Tests for the generated-text style evaluation."""

import pytest
import torch
from torch import nn

from specialk.trainers.evaluate_style_transfer import (
    corpus_bleu,
    load_style_weights,
    style_accuracy,
)


def test_style_accuracy_counts_the_target_label():
    predictions = torch.tensor([1, 1, 0, 1])

    assert style_accuracy(predictions, target_label=1) == 0.75
    assert style_accuracy(predictions, target_label=0) == 0.25


def test_corpus_bleu_is_100_for_identical_texts():
    references = ["the senator voted against the bill", "we need new taxes"]
    hypotheses = list(references)

    assert corpus_bleu(hypotheses, references) == pytest.approx(100.0)


def test_corpus_bleu_drops_when_the_output_differs():
    references = ["the senator voted against the bill"]
    perfect = corpus_bleu(references, references)
    worse = corpus_bleu(["the senator voted for the bill"], references)

    assert perfect is not None and worse is not None
    assert worse < perfect


def test_corpus_bleu_without_references_is_none():
    assert corpus_bleu(["anything"], []) is None


def test_load_style_weights_reads_the_sequence_model_prefix(tmp_path):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2))

    model = Tiny()
    checkpoint = tmp_path / "style.ckpt"
    torch.save(
        {
            "state_dict": {
                "sequence_model.weight": torch.tensor([1.0, 2.0]),
                "classifier.weight": torch.tensor([9.0]),
            }
        },
        checkpoint,
    )

    load_style_weights(model, checkpoint)

    assert torch.equal(model.weight.detach(), torch.tensor([1.0, 2.0]))
