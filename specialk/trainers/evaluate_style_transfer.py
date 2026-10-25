"""Measure style transfer on generated translations.

Training logs an accuracy on teacher-forced targets, which says how well the
decoder fits its training text, not how its own output reads. This entry point
generates translations from French sources, runs the frozen classifier on the
generated English, and reports style accuracy and BLEU for the base checkpoint
and the style-tuned one.

Example:
    python -m specialk.trainers.evaluate_style_transfer \\
        --mt-checkpoint checkpoints/mt.ckpt \\
        --style-checkpoint checkpoints/style.ckpt \\
        --classifier-checkpoint checkpoints/cnn.ckpt \\
        --dataset political --target-label 1 --limit 200
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import sacrebleu
import torch
from torch import Tensor

from specialk.core.utils import batch_texts, check_torch_device, log
from specialk.models.classifier.models import BERTClassifier, CNNClassifier
from specialk.models.tokenizer import Vocabulary, WordVocabulary
from specialk.trainers.common import load_style_datasets
from specialk.trainers.train_style_decoder import (
    DATASETS,
    load_classifier,
    load_marian,
    load_seq2seq,
)

SEQ2SEQ = ("transformer", "marian")


# ----------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------
def generate_texts(
    model: Any,
    sources: list[str],
    batch_size: int,
    max_new_tokens: int,
    seq2seq: str,
    no_repeat_ngram_size: int = 0,
) -> list[str]:
    """Translate ``sources`` with the model as-is."""
    generated: list[str] = []
    device = model.device

    for batch in batch_texts(sources, batch_size):
        if seq2seq == "marian":
            encoded = model.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=model.sequence_length,
            ).to(device)
            with torch.inference_mode():
                output = model.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    no_repeat_ngram_size=no_repeat_ngram_size or None,
                )
            generated.extend(
                model.tokenizer.batch_decode(output, skip_special_tokens=True)
            )
        else:
            tokens = model.tokenizer.to_tensor(batch)
            if tokens.dim() == 3:
                tokens = tokens.squeeze(1)
            tokens = tokens.to(device)
            output = model.generate(tokens, max_len=max_new_tokens, temperature=0)
            generated.extend(model.decoder_tokenizer.detokenize(output, specials=False))

    return generated


# ----------------------------------------------------------------------
# Classification of generated text
# ----------------------------------------------------------------------
def classifier_tokenizer(
    args: argparse.Namespace,
    mt_model: Any,
    classifier: CNNClassifier | BERTClassifier,
) -> Vocabulary | None:
    """The tokenizer the CNN was trained with, or ``None`` for BERT."""
    if isinstance(classifier, BERTClassifier):
        return None
    if classifier.tokenizer is not None:
        return classifier.tokenizer
    if args.classifier_tokenizer is not None:
        return WordVocabulary.from_file(args.classifier_tokenizer)

    tokenizer = mt_model.tokenizer
    if tokenizer.vocab_size != classifier.vocabulary_size:
        raise ValueError(
            f"The classifier expects {classifier.vocabulary_size} tokens but the "
            f"seq2seq tokenizer has {tokenizer.vocab_size}; pass "
            "--classifier-tokenizer with the tokenizer the classifier was trained on"
        )
    return tokenizer


def classify_texts(
    classifier: CNNClassifier | BERTClassifier,
    tokenizer: Vocabulary | None,
    texts: list[str],
    target_label: int,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    """Return predictions and the probability of ``target_label`` per text."""
    device = classifier.device

    if isinstance(classifier, CNNClassifier):
        if tokenizer is None:
            raise ValueError("The CNN classifier needs a tokenizer")
        scores = []
        for batch in batch_texts(texts, batch_size):
            tokens = tokenizer.to_tensor(batch)
            if tokens.dim() == 3:
                tokens = tokens.squeeze(1)
            tokens = tokens.to(device)
            one_hot = classifier.one_hot(tokens)
            with torch.inference_mode():
                scores.append(torch.sigmoid(classifier.model(one_hot).reshape(-1)))
        positive = torch.cat(scores).float().cpu()
        threshold = getattr(classifier, "decision_threshold", 0.5)
        predictions = (positive > threshold).long()
        target = positive if target_label == 1 else 1 - positive
        return predictions, target

    batches = []
    for batch in batch_texts(texts, batch_size):
        encoded = classifier.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=classifier.sequence_length,
        ).to(device)
        with torch.inference_mode():
            logits = classifier.model(**encoded).logits
        batches.append(torch.softmax(logits.float(), dim=-1).cpu())
    probabilities = torch.cat(batches)
    return probabilities.argmax(dim=-1), probabilities[:, target_label]


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def style_accuracy(predictions: Tensor, target_label: int) -> float:
    return float((predictions == target_label).float().mean())


def corpus_bleu(hypotheses: list[str], references: list[str]) -> float | None:
    """Corpus BLEU, case-folded because the political references are lowercase.

    The references also come from the old corpus with spaces before punctuation,
    so let SacreBLEU's 13a tokenizer normalise both sides.
    """
    if not references:
        return None
    score = sacrebleu.corpus_bleu(hypotheses, [references], lowercase=True).score
    return float(score)


def repetition_stats(texts: list[str]) -> tuple[float, float]:
    """Return mean token repetition and the share of badly looping rows.

    A token-level ratio near zero means ordinary prose. A ratio above 0.5 means
    the decoder is repeating itself, which style accuracy alone will not show.
    """
    ratios = []
    for text in texts:
        tokens = text.split()
        ratios.append(0.0 if not tokens else 1.0 - len(set(tokens)) / len(tokens))
    if not ratios:
        return 0.0, 0.0
    mean = sum(ratios) / len(ratios)
    share = sum(ratio > 0.5 for ratio in ratios) / len(ratios)
    return mean, share


def evaluate(
    model: Any,
    classifier: CNNClassifier | BERTClassifier,
    clf_tokenizer: Vocabulary | None,
    sources: list[str],
    references: list[str],
    target_label: int,
    batch_size: int,
    max_new_tokens: int,
    seq2seq: str,
    samples: int = 3,
    no_repeat_ngram_size: int = 0,
) -> dict:
    """Generate, classify and score one model."""
    generated = generate_texts(
        model,
        sources,
        batch_size,
        max_new_tokens,
        seq2seq,
        no_repeat_ngram_size,
    )
    predictions, probability = classify_texts(
        classifier, clf_tokenizer, generated, target_label, batch_size
    )
    repetition, degenerate = repetition_stats(generated)

    return {
        "rows": len(generated),
        "style_accuracy": style_accuracy(predictions, target_label),
        "mean_target_probability": float(probability.mean()),
        "bleu": corpus_bleu(generated, references),
        "token_repetition": repetition,
        "degenerate_rows": degenerate,
        "samples": [
            {
                "source": source,
                "reference": reference,
                "generated": hypothesis,
                "target_probability": float(score),
                "predicted_label": int(prediction),
            }
            for source, reference, hypothesis, score, prediction in zip(
                sources[:samples],
                references[:samples],
                generated[:samples],
                probability[:samples],
                predictions[:samples],
            )
        ],
    }


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_style_weights(model: Any, checkpoint_path: Path) -> None:
    """Copy the fine-tuned seq2seq weights from a style-decoder checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    prefix = "sequence_model."
    seq_state = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not seq_state:
        raise ValueError(
            f"{checkpoint_path} has no 'sequence_model.' weights; "
            "pass a checkpoint written by train_style_decoder"
        )
    missing, unexpected = model.load_state_dict(seq_state, strict=False)
    log.info(
        "Loaded the style-tuned weights",
        checkpoint=str(checkpoint_path),
        missing=len(missing),
        unexpected=len(unexpected),
    )


def base_model(args: argparse.Namespace) -> Any:
    if args.seq2seq == "marian":
        return load_marian(args.mt_checkpoint, None)
    return load_seq2seq(args.mt_checkpoint)


def styled_model(args: argparse.Namespace) -> Any:
    if args.seq2seq == "marian":
        model = load_marian(args.mt_checkpoint, args.lora_r)
    else:
        model = load_seq2seq(args.mt_checkpoint)
    load_style_weights(model, args.style_checkpoint)
    return model


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate translations with a base and a style-tuned seq2seq, "
            "then score the generated English with the frozen classifier."
        )
    )
    parser.add_argument("--dataset", choices=DATASETS, default="political")
    parser.add_argument("--path", type=Path, default=None)
    parser.add_argument("--eval-path", type=Path, default=None)
    parser.add_argument("--source-column", default=None)
    parser.add_argument("--target-label", type=int, default=1)
    parser.add_argument(
        "--eval-label",
        type=int,
        default=None,
        help="only evaluate rows whose reference has this label",
    )
    parser.add_argument("--mt-checkpoint", type=Path, required=True)
    parser.add_argument("--seq2seq", choices=SEQ2SEQ, default="transformer")
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--style-checkpoint", type=Path, default=None)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--classifier", choices=("cnn", "bert"), default="cnn")
    parser.add_argument(
        "--classifier-threshold",
        type=float,
        default=None,
        help="override the CNN's sigmoid decision threshold",
    )
    parser.add_argument("--classifier-tokenizer", type=Path, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="number of evaluation rows (0 means all)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=0,
        help="block repeated n-grams during generation (0 disables it)",
    )
    parser.add_argument(
        "--accelerator", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    device = check_torch_device() if args.accelerator == "auto" else args.accelerator

    _, evaluation = load_style_datasets(
        args.dataset,
        target_label=None,
        path=args.path,
        source_column=args.source_column,
        eval_path=args.eval_path,
    )
    if args.eval_label is not None:
        evaluation = evaluation.filter(lambda row: row["label"] == args.eval_label)
    if args.limit:
        evaluation = evaluation.select(range(min(args.limit, len(evaluation))))
    sources = list(evaluation["source"])
    references = list(evaluation["target"])
    log.info("Loaded the evaluation set", rows=len(sources))

    classifier = load_classifier(
        args.classifier, args.classifier_checkpoint, args.classifier_tokenizer
    )
    if args.classifier_threshold is not None:
        classifier.decision_threshold = args.classifier_threshold
    classifier.to(device).eval()

    model = base_model(args).to(device).eval()
    clf_tokenizer = classifier_tokenizer(args, model, classifier)

    results: dict[str, Any] = {}
    results["base"] = evaluate(
        model,
        classifier,
        clf_tokenizer,
        sources,
        references,
        args.target_label,
        args.batch_size,
        args.max_new_tokens,
        args.seq2seq,
        args.samples,
        args.no_repeat_ngram_size,
    )

    if args.style_checkpoint is not None:
        model = styled_model(args).to(device).eval()
        results["styled"] = evaluate(
            model,
            classifier,
            clf_tokenizer,
            sources,
            references,
            args.target_label,
            args.batch_size,
            args.max_new_tokens,
            args.seq2seq,
            args.samples,
            args.no_repeat_ngram_size,
        )

    results["settings"] = {
        "dataset": args.dataset,
        "rows": len(sources),
        "target_label": args.target_label,
        "eval_label": args.eval_label,
        "mt_checkpoint": str(args.mt_checkpoint),
        "style_checkpoint": str(args.style_checkpoint)
        if args.style_checkpoint
        else None,
        "classifier_checkpoint": str(args.classifier_checkpoint),
    }

    text = json.dumps(results, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return results


if __name__ == "__main__":
    main()
