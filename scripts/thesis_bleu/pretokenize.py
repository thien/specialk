#!/usr/bin/env python
"""Pre-tokenise raw text with the Moses settings used by the 2019 models."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from specialk.models.tokenizers.moses import MosesPreTokenizer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--lang", required=True, choices=["en", "fr"])
    parser.add_argument("--no-lower", action="store_true")
    args = parser.parse_args()

    tokenize = MosesPreTokenizer(args.lang, lower=not args.no_lower)
    lines = args.src.read_text(encoding="utf-8").splitlines()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for line in lines:
            out.write(tokenize(line) + "\n")
    print(f"[info] wrote {len(lines)} lines to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
