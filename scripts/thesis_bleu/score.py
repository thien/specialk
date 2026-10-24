#!/usr/bin/env python
"""Score translation hypotheses the way the thesis did, plus sacreBLEU.

The thesis used Maluuba nlg-eval, which wraps pycocoevalcap's coco-caption
BLEU.  That scorer splits on whitespace and computes cumulative BLEU-1..4.
Hypotheses and references here are Moses-tokenized text, so feeding them
directly to the coco scorer reproduces the thesis numbers.

For the modern number the two sides are detokenized with sacremoses and
scored with sacreBLEU (default 13a tokenizer).

Usage:
    python scripts/thesis_bleu/score.py --hyp H --ref R --lang fr
"""

import argparse
import sys
from pathlib import Path

from pycocoevalcap.bleu.bleu import Bleu
from sacremoses import MosesDetokenizer


def read_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def coco_bleu(hyps: list[str], refs: list[str]) -> list[float]:
    if len(hyps) != len(refs):
        raise SystemExit(
            f"line count mismatch: {len(hyps)} hypotheses vs {len(refs)} references"
        )
    gts = {i: [ref] for i, ref in enumerate(refs)}
    res = {i: [hyp] for i, hyp in enumerate(hyps)}
    score, _ = Bleu(4).compute_score(gts, res, verbose=0)
    return [value * 100 for value in score]


def detokenize(lines: list[str], lang: str) -> list[str]:
    detok = MosesDetokenizer(lang=lang)
    # The corpus was tokenized with -no-escape, so do not unescape entities.
    return [detok.detokenize(line.split(), unescape=False) for line in lines]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hyp", required=True, type=Path)
    parser.add_argument("--ref", required=True, type=Path)
    parser.add_argument(
        "--lang",
        required=True,
        choices=["en", "fr"],
        help="target language, used for sacremoses detokenization",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="only score the first N lines"
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    hyps = read_lines(args.hyp)
    refs = read_lines(args.ref)
    if args.limit:
        hyps, refs = hyps[: args.limit], refs[: args.limit]

    coco = coco_bleu(hyps, refs)

    sacre_hyps = detokenize(hyps, args.lang)
    sacre_refs = detokenize(refs, args.lang)

    import sacrebleu

    metric = sacrebleu.BLEU()
    bleu = metric.corpus_score(sacre_hyps, [sacre_refs])
    result = {
        "lines": len(hyps),
        "coco_bleu": {f"BLEU_{i + 1}": round(value, 4) for i, value in enumerate(coco)},
        "sacrebleu": round(bleu.score, 4),
        "sacrebleu_signature": str(metric.get_signature()),
    }

    if args.json:
        import json

        print(json.dumps(result, indent=2))
    else:
        print(f"lines: {result['lines']}")
        print(
            "coco BLEU-1..4: "
            + " / ".join(f"{value:.4f}" for value in result["coco_bleu"].values())
        )
        print(f"sacreBLEU: {result['sacrebleu']:.4f}")
        print(f"signature: {result['sacrebleu_signature']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
