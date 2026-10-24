"""Generate translations from a trained checkpoint.

Replaces the legacy Fairseq-era script: one self-contained checkpoint is enough,
because the tokenizers and the architecture are stored inside it. See
``doc/generation.md`` for the pipeline this belongs to.

Example:
    python -m specialk.translate --checkpoint model.ckpt \\
        --src input.en --output predictions.fr
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
from tqdm import tqdm

from specialk.core.utils import check_torch_device
from specialk.models.mt_model import NMTModule, RNNModule
from specialk.models.transformer.torch.pytorch_transformer import (
    PyTorchTransformerModule,
)

DEVICES = ("mps", "cuda", "cpu")


def load_module(checkpoint: Path | str, device: str = "cpu") -> NMTModule:
    """Load a checkpoint in eval mode, picking the module class from its hyperparameters."""
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    # Read only the hyperparameters to decide which LightningModule to build.
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hparams = raw.get("hyper_parameters", {})
    module_class = RNNModule if "rnn_size" in hparams else PyTorchTransformerModule
    del raw  # don't keep a second copy of the weights while loading

    module = module_class.load_from_checkpoint(
        checkpoint, map_location=device, weights_only=False
    )
    if module.tokenizer is None or module.decoder_tokenizer is None:
        raise ValueError(
            f"{checkpoint} has no tokenizers; retrain it with "
            "specialk.trainers.train so generation works"
        )

    # Inference only: on MPS, train-mode dropout is rejected by
    # scaled_dot_product_attention.
    return module.eval().to(device)


def resolve_device(requested: str | None) -> str:
    """Return a usable device, preferring MPS (then CUDA) when not given."""
    device = requested or check_torch_device()
    device_type = torch.device(device).type
    if device_type not in DEVICES:
        raise ValueError(
            f"unsupported device {device!r}; expected one of {', '.join(DEVICES)}"
        )
    if device_type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def read_lines(path: Path | str) -> list[str]:
    with open(path, encoding="utf-8") as src:
        return [line.rstrip("\n") for line in src]


def translate(
    module: NMTModule,
    lines: Sequence[str],
    *,
    beam_size: int = 5,
    n_best: int = 1,
    batch_size: int = 128,
    max_new_tokens: int = 100,
    no_repeat_ngram_size: int | None = None,
    device: str = "cpu",
) -> list[str]:
    """Translate lines, returning ``n_best`` hypotheses per input line (best first)."""
    if n_best > beam_size:
        raise ValueError(
            f"n_best (currently {n_best}) must be <= beam_size (currently {beam_size})"
        )

    hypotheses: list[str] = []
    batches = range(0, len(lines), batch_size)
    for start in tqdm(batches, disable=len(lines) <= batch_size):
        batch = list(lines[start : start + batch_size])
        tokens = module.tokenizer.to_tensor(batch).to(device)
        _, output = module.beam_search(
            tokens,
            num_return_sequences=n_best,
            num_beams=beam_size,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        # batch x n_best x seq_len, flattened in input-major order.
        flat = output.reshape(-1, output.size(-1))
        decoded = module.decoder_tokenizer.detokenize(flat, specials=False)
        # Single-row inputs come back as a str for some tokenizers.
        if isinstance(decoded, str):
            decoded = [decoded]
        hypotheses.extend(decoded)

    return hypotheses


def write_predictions(path: Path | str, hypotheses: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        out.writelines(f"{hypothesis.strip()}\n" for hypothesis in hypotheses)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate translations from a trained checkpoint.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Lightning .ckpt to load."
    )
    parser.add_argument(
        "--src", type=Path, required=True, help="Input text, one sentence per line."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions.txt"),
        help="Output file (n-best lines per input line).",
    )
    parser.add_argument("--beam-size", type=int, default=5, help="Number of beams.")
    parser.add_argument(
        "--n-best",
        type=int,
        default=1,
        help="Hypotheses kept per input; must be <= --beam-size.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Sentences per forward pass."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Generation length cap.",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=None,
        help="Block repeated n-grams of this size.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="mps, cuda, cuda:N or cpu (default: auto-detected, MPS first).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> list[str]:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.n_best > args.beam_size:
        parser.error("--n-best must be <= --beam-size")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    device = resolve_device(args.device)
    module = load_module(args.checkpoint, device=device)
    lines = read_lines(args.src)
    hypotheses = translate(
        module,
        lines,
        beam_size=args.beam_size,
        n_best=args.n_best,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        device=device,
    )
    write_predictions(args.output, hypotheses)
    return hypotheses


if __name__ == "__main__":
    main()
