"""Score a CNN classifier on a labelled text file or the old tokenized split.

    python3 scripts/classifier/evaluate_classifier.py CHECKPOINT \
        --tokenized political.train.pt --split valid

CHECKPOINT can be a run directory, in which case every ``epoch=*.ckpt`` is
scored and the best one is printed. The legacy checkpoints need their sibling
``hyperparameters.yaml``; the script falls back to it automatically.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from specialk.models.classifier.models import CNNClassifier
from specialk.models.tokenizer import WordVocabulary
from specialk.trainers.train_classifier import (
    load_legacy_tokenized_data,
    load_text_classification_dataset,
)


def load_model(path: Path, device: str) -> CNNClassifier:
    guesses = [None]
    sibling = path.parent / "hyperparameters.yaml"
    if sibling.exists():
        guesses.append(sibling)
    last_error = None
    for hparams in guesses:
        try:
            model = CNNClassifier.load_from_checkpoint(
                path, hparams_file=hparams, map_location=device
            )
            return model.eval().to(device)
        except Exception as error:  # noqa: BLE001 - try the next loading style
            last_error = error
    raise last_error


def batch_probabilities(
    model: CNNClassifier, rows: list[torch.Tensor], batch_size: int
) -> torch.Tensor:
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chunk = torch.stack(rows[start : start + batch_size]).long()
            outputs.append(model.model(model.one_hot(chunk)).squeeze(-1))
    return torch.cat(outputs).cpu()


def text_rows(
    model: CNNClassifier,
    text_path: Path,
    positive_label: str,
) -> tuple[list[torch.Tensor], list[int]]:
    if model.tokenizer is None:
        raise SystemExit("--text needs a checkpoint or --tokenizer with a vocabulary")
    dataset, _ = load_text_classification_dataset(
        text_path, text_path, positive_label
    )
    tokens = model.tokenizer.to_tensor(list(dataset["text"]))
    rows = [token for token in tokens]
    return rows, list(dataset["label"])


def tokenized_rows(
    tokenized_path: Path, split: str, positive_label: str
) -> tuple[list[torch.Tensor], list[int]]:
    data = load_legacy_tokenized_data(tokenized_path)
    rows = [row.long() for row in data[split]["src"]]
    labels = [int(label) for label in data[split]["tgt"]]
    if positive_label == "democratic":
        labels = [1 - label for label in labels]
    return rows, labels


def score(
    model: CNNClassifier,
    rows: list[torch.Tensor],
    labels: list[int],
    batch_size: int,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    probabilities = batch_probabilities(model, rows, batch_size)
    targets = torch.tensor(labels)
    predictions = (probabilities > 0.5).long()
    accuracy = (predictions == targets).float().mean().item()
    return accuracy, probabilities, predictions


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="a .ckpt or a run directory")
    parser.add_argument("--tokenized", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "valid"], default="valid")
    parser.add_argument("--text", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument(
        "--positive-label", choices=["republican", "democratic"], default="republican"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    checkpoints = (
        sorted(
            args.checkpoint.glob("epoch=*.ckpt"),
            key=lambda path: int(path.stem.split("-")[0].split("=")[1]),
        )
        if args.checkpoint.is_dir()
        else [args.checkpoint]
    )
    if not checkpoints:
        raise SystemExit(f"no epoch checkpoints in {args.checkpoint}")

    model = load_model(checkpoints[0], args.device)
    if args.tokenizer:
        model.tokenizer = WordVocabulary.from_file(args.tokenizer)

    if args.tokenized:
        rows, labels = tokenized_rows(
            args.tokenized, args.split, args.positive_label
        )
    elif args.text:
        rows, labels = text_rows(model, args.text, args.positive_label)
    else:
        raise SystemExit("pass --tokenized or --text")

    if len(checkpoints) > 1:
        best_accuracy = -1.0
        best_checkpoint = None
        for checkpoint in checkpoints:
            model = load_model(checkpoint, args.device)
            if args.tokenizer:
                model.tokenizer = WordVocabulary.from_file(args.tokenizer)
            accuracy, _, _ = score(model, rows, labels, args.batch_size)
            print(f"{checkpoint.stem:<24} {accuracy:.4f}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_checkpoint = checkpoint
        print(f"{'best':<24} {best_accuracy:.4f}  {best_checkpoint.name}")
        return 0

    accuracy, probabilities, predictions = score(
        model, rows, labels, args.batch_size
    )
    targets = torch.tensor(labels)

    print(f"checkpoint  {args.checkpoint}")
    print(f"examples    {len(labels)}")
    print(f"accuracy    {accuracy:.4f}")
    print(f"mean prob   {probabilities.mean().item():.4f}")
    for name, value in (("positive", 1), ("negative", 0)):
        mask = targets == value
        recall = (predictions[mask] == value).float().mean().item()
        print(f"recall {name:<7} {recall:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
