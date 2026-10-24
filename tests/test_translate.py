from pathlib import Path

import pytest
import torch
from lightning.pytorch import Trainer

from specialk import translate
from specialk.models.mt_model import RNNModule
from specialk.models.tokenizer import SentencePieceVocabulary
from specialk.models.transformer.torch.pytorch_transformer import (
    PyTorchTransformerModule,
)

TOKENIZER_PATH = Path("assets/tokenizer/sentencepiece/enfr_small.model")
SEQUENCE_LENGTH = 12
SOURCE_LINES = [
    "Ein Mann schläft auf einem Sofa.",
    "Eine Gruppe von Männern lädt Baumwolle auf einen Lastwagen.",
    "Hallo.",
]
DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.fixture(scope="module")
def tokenizer() -> SentencePieceVocabulary:
    return SentencePieceVocabulary.from_file(TOKENIZER_PATH, max_length=SEQUENCE_LENGTH)


def save_checkpoint(module, path: Path) -> Path:
    """Write a Lightning checkpoint without training (random weights are fine)."""
    trainer = Trainer(logger=False, accelerator="cpu")
    trainer.strategy.connect(module)
    trainer.save_checkpoint(path)
    return path


@pytest.fixture(scope="module")
def transformer_checkpoint(tmp_path_factory, tokenizer):
    torch.manual_seed(0)
    module = PyTorchTransformerModule(
        name="test",
        vocabulary_size=tokenizer.vocab_size,
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=tokenizer,
        dim_model=16,
        n_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout=0.0,
    )
    path = tmp_path_factory.mktemp("checkpoints") / "transformer.ckpt"
    return save_checkpoint(module, path)


@pytest.fixture(scope="module")
def rnn_checkpoint(tmp_path_factory, tokenizer):
    torch.manual_seed(0)
    module = RNNModule(
        name="test",
        vocabulary_size=tokenizer.vocab_size,
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=tokenizer,
        rnn_size=16,
        d_word_vec=8,
        layers=1,
        brnn=False,
        dropout=0.0,
    )
    path = tmp_path_factory.mktemp("checkpoints") / "rnn.ckpt"
    return save_checkpoint(module, path)


@pytest.mark.parametrize("checkpoint", ["transformer_checkpoint", "rnn_checkpoint"])
def test_translate_writes_n_best_lines(request, checkpoint, tmp_path):
    checkpoint = request.getfixturevalue(checkpoint)
    src = tmp_path / "src.txt"
    src.write_text("\n".join(SOURCE_LINES) + "\n", encoding="utf-8")
    output = tmp_path / "predictions.txt"

    n_best = 2
    hypotheses = translate.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--src",
            str(src),
            "--output",
            str(output),
            "--beam-size",
            "3",
            "--n-best",
            str(n_best),
            "--batch-size",
            "2",
            "--max-new-tokens",
            "5",
            "--device",
            "cpu",
        ]
    )

    assert len(hypotheses) == len(SOURCE_LINES) * n_best
    assert len(output.read_text(encoding="utf-8").splitlines()) == len(hypotheses)
    assert all(hypothesis.strip() for hypothesis in hypotheses)


@pytest.mark.parametrize("device", DEVICES)
def test_rnn_beam_search_returns_ranked_hypotheses(tokenizer, device):
    module = RNNModule(
        name="test",
        vocabulary_size=tokenizer.vocab_size,
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=tokenizer,
        rnn_size=16,
        d_word_vec=8,
        layers=1,
        brnn=False,
        dropout=0.0,
    ).eval()
    module.to(device)

    tokens = tokenizer.to_tensor(SOURCE_LINES[:2]).to(device)
    n_best, n_beams = 2, 3
    scores, hypotheses = module.beam_search(
        tokens,
        num_return_sequences=n_best,
        num_beams=n_beams,
        max_new_tokens=5,
        no_repeat_ngram_size=2,
    )

    assert scores.shape == (len(SOURCE_LINES[:2]), n_best)
    assert hypotheses.shape[:2] == (len(SOURCE_LINES[:2]), n_best)
    assert torch.all(scores[:, 0] >= scores[:, 1])
