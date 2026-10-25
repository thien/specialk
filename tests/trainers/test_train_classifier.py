"""Smoke test for the classifier trainer CLI and checkpoint reload."""

import pytest
import torch

from datasets import Dataset, DatasetDict, Features, Value
from specialk.models.classifier.models import CNNClassifier
from specialk.trainers import train_classifier

VOCAB_SIZE = 32
SEQUENCE_LENGTH = 8
LEARNING_RATE = 0.007


class TinyTokenizer:
    """Stand-in for a real vocabulary, small enough to run in a second."""

    vocab_size = VOCAB_SIZE
    max_length = SEQUENCE_LENGTH
    PAD = 0

    def to_tensor(self, texts):
        return [
            [1 + (len(text) + i) % (VOCAB_SIZE - 1) for i in range(self.max_length)]
            for text in texts
        ]


def tiny_dataset() -> DatasetDict:
    features = Features({"text": Value("string"), "label": Value("int64")})
    train = Dataset.from_dict(
        {"text": [f"row {i}" for i in range(6)], "label": [0, 1, 0, 1, 0, 1]},
        features=features,
    )
    valid = Dataset.from_dict(
        {"text": ["a", "b"], "label": [0, 1]}, features=features
    )
    return DatasetDict({"train": train, "eval": valid})


@pytest.mark.lightweight
def test_cli_smoke_and_checkpoint_reload(monkeypatch, tmp_path):
    tokenizer = TinyTokenizer()
    monkeypatch.setattr(train_classifier, "get_tokenizer", lambda *a, **k: tokenizer)
    monkeypatch.setattr(train_classifier, "load_dataset", lambda *a, **k: tiny_dataset())
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.chdir(tmp_path)

    # Keep the real loader but skip the eight worker processes.
    real_init_dataloader = train_classifier.init_dataloader

    def init_dataloader(dataset, tokenizer, batch_size, shuffle, **kwargs):
        return real_init_dataloader(
            dataset,
            tokenizer,
            batch_size,
            shuffle=shuffle,
            n_workers=0,
            persistent_workers=False,
        )

    monkeypatch.setattr(train_classifier, "init_dataloader", init_dataloader)

    checkpoint_dir = tmp_path / "checkpoints"
    base_args = [
        "--model_type",
        "cnn",
        "--batch_size",
        "2",
        "--epochs",
        "1",
        "--lr",
        str(LEARNING_RATE),
        "--accelerator",
        "auto",
        "--limit-train-batches",
        "2",
        "--limit-val-batches",
        "1",
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    args = train_classifier.get_args(base_args)
    trainer = train_classifier.main_new(args)

    assert trainer.global_step == 2
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    assert checkpoints, "the trainer did not write a checkpoint"

    # No hparams file: everything has to come from the checkpoint itself.
    module = CNNClassifier.load_from_checkpoint(checkpoints[0], map_location="cpu")
    assert module.vocabulary_size == VOCAB_SIZE
    assert module.sequence_length == SEQUENCE_LENGTH
    assert module.hparams.learning_rate == pytest.approx(LEARNING_RATE)
    assert module.tokenizer is None

    module.eval()
    with torch.inference_mode():
        output = module(torch.zeros(SEQUENCE_LENGTH, 2, VOCAB_SIZE))
    assert output.shape == (2, 1)

    resumed = train_classifier.get_args([*base_args, "--resume", "--epochs", "2"])
    trainer = train_classifier.main_new(resumed)
    assert trainer.global_step == 4


@pytest.mark.lightweight
def test_local_text_dataset_and_sgd_schedule(tmp_path):
    train = tmp_path / "train.txt"
    valid = tmp_path / "valid.txt"
    train.write_text("democratic one\nrepublican two\n")
    valid.write_text("democratic three\n")

    train_dataset, valid_dataset = train_classifier.load_text_classification_dataset(
        train, valid, "republican"
    )
    assert train_dataset["label"] == [0, 1]
    assert valid_dataset["label"] == [0]

    module = CNNClassifier(
        "parity",
        vocabulary_size=VOCAB_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        learning_rate=1.0,
        optimizer="sgd",
        lr_decay=0.5,
        decay_start=8,
    )
    config = module.configure_optimizers()
    assert isinstance(config["optimizer"], torch.optim.SGD)
    scheduler = config["lr_scheduler"]["scheduler"]
    for _ in range(8):
        config["optimizer"].step()
        scheduler.step()
    assert config["optimizer"].param_groups[0]["lr"] == pytest.approx(0.5)


@pytest.mark.lightweight
def test_legacy_tokenized_labels_are_flipped(monkeypatch):
    import io

    from specialk.models.tokenizers.dict import Dict

    unpickler = train_classifier._LegacyDictUnpickler(io.BytesIO(b""))
    assert unpickler.find_class("onmt", "Dict") is Dict

    class FakeDict:
        def size(self):
            return VOCAB_SIZE

    data = {
        "dicts": {"src": FakeDict()},
        "train": {
            "src": [torch.arange(4), torch.arange(4) + 1],
            "tgt": [torch.tensor([1]), torch.tensor([0])],
        },
        "valid": {"src": [torch.arange(4)], "tgt": [torch.tensor([0])]},
    }
    monkeypatch.setattr(
        train_classifier, "load_legacy_tokenized_data", lambda path: data
    )

    train, valid, sequence_length, vocabulary_size = (
        train_classifier.load_tokenized_classification_dataset(
            "political.train.pt", batch_size=2, positive_label="democratic"
        )
    )
    assert sequence_length == 4
    assert vocabulary_size == VOCAB_SIZE
    assert train.dataset.labels == [0, 1]
    assert valid.dataset.labels == [1]


@pytest.mark.lightweight
def test_resolve_batch_size_defaults():
    assert train_classifier.resolve_batch_size("cnn", None) == 128
    assert train_classifier.resolve_batch_size("bert", None) == 32
    assert train_classifier.resolve_batch_size("bert", 8) == 8


@pytest.mark.lightweight
@pytest.mark.parametrize(
    "flag", ["--tokenized-path", "--tokenizer-path", "--mt-checkpoint"]
)
def test_bert_rejects_cnn_tokenizers(flag):
    with pytest.raises(SystemExit):
        train_classifier.get_args(["--model_type", "bert", flag, "some-path"])


@pytest.mark.lightweight
def test_political_labels_match_the_style_dataset():
    dataset = Dataset.from_dict(
        {"text": ["a", "b"], "label": ["democratic", "republican"]}
    )

    encoded = train_classifier._encode_labels(dataset, "thien/political")

    assert encoded["label"] == [1, 0]


@pytest.mark.lightweight
def test_unknown_political_label_is_rejected():
    dataset = Dataset.from_dict({"text": ["a"], "label": ["green"]})

    with pytest.raises(ValueError, match="Unknown political labels"):
        train_classifier._encode_labels(dataset, "thien/political")
