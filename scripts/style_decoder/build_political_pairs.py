#!/usr/bin/env python
"""Build French-English political pairs from the local corpus.

``datasets/political_data`` holds the English political sentences and the
French files that ``scripts/scripts/setup_political_dataset.sh`` produced with
an en-to-fr model. The English lines start with the party name; the style
decoder wants that party as an integer label.

Example:
    python scripts/style_decoder/build_political_pairs.py
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd
from sacremoses import MosesDetokenizer

LABELS = {"democratic": 1, "republican": 0}


def _clean(text: str, detokenizer: MosesDetokenizer | None) -> str:
    text = html.unescape(text.strip())
    if detokenizer is None:
        return text
    return detokenizer.detokenize(text.split())


def read_pairs(
    political_dir: Path,
    party: str,
    split: str,
    french_detokenizer: MosesDetokenizer | None = None,
    english_detokenizer: MosesDetokenizer | None = None,
) -> list[tuple[str, str, int]]:
    french = (political_dir / f"{party}_only.{split}.en.ts").read_text(encoding="utf-8")
    english = (political_dir / f"{party}_only.{split}.en").read_text(encoding="utf-8")
    french_lines = french.splitlines()
    english_lines = english.splitlines()
    if len(french_lines) != len(english_lines):
        raise ValueError(
            f"{party} {split}: {len(french_lines)} French lines but "
            f"{len(english_lines)} English lines"
        )

    rows = []
    for source, target in zip(french_lines, english_lines):
        target = target.strip()
        if not target:
            continue
        prefix, _, body = target.partition(" ")
        if prefix not in LABELS:
            raise ValueError(f"Unexpected line prefix {prefix!r} in {party} {split}")
        rows.append(
            (
                _clean(source, french_detokenizer),
                _clean(body, english_detokenizer),
                LABELS[prefix],
            )
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--political-dir",
        type=Path,
        default=Path("datasets/political_data"),
        help="directory with the *_only.<split>.en and .en.ts files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/political_pairs"),
        help="where to write train.parquet and dev.parquet",
    )
    parser.add_argument(
        "--keep-tokenized",
        action="store_true",
        help="keep the Moses tokenization instead of detokenizing both sides",
    )
    args = parser.parse_args(argv)

    french_detokenizer = None if args.keep_tokenized else MosesDetokenizer(lang="fr")
    english_detokenizer = None if args.keep_tokenized else MosesDetokenizer(lang="en")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        rows = read_pairs(
            args.political_dir,
            "democratic",
            split,
            french_detokenizer,
            english_detokenizer,
        )
        rows += read_pairs(
            args.political_dir,
            "republican",
            split,
            french_detokenizer,
            english_detokenizer,
        )
        frame = pd.DataFrame(rows, columns=["text_fr", "text", "label"])
        path = args.output_dir / f"{split}.parquet"
        frame.to_parquet(path, index=False)
        print(f"{path}: {len(frame)} rows")


if __name__ == "__main__":
    main()
