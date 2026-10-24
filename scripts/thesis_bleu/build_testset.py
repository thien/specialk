#!/usr/bin/env python
"""Build the held-out test set used for the thesis BLEU re-run.

Applies the filtering logic from
``datasets/machine_translation/filter_gm_dataset.ipynb`` verbatim to the
Moses-tokenized ``corpus_enfr_final.test.{en,fr}`` pair, then draws fixed-seed
5K and 20K subsets.

The notebook's output writes were commented out, so this script is the only
place the filtered *test* split is materialised.
"""

import argparse
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from specialk.models.tokenizers.bpe import Encoder as BPE  # noqa: E402

BAD_SUBSTRINGS = [
    "www",
    "´",
    "Ã",
    "®",
    "©",
    "|",
    "<",
    ">",
    "^",
    "url",
    "http",
    "whuqhv",
    "( aaaa / mm / jj )",
    "ة",
    "ي",
]

LIMITS = {".": 5, "(": 3, ")": 3, ",": 6, "@": 5, "_": 3}


def verify(fr: str) -> bool:
    """Port of the notebook's verify().  ``fr`` includes its newline."""
    if any(bad in fr for bad in BAD_SUBSTRINGS):
        return False
    counts: dict[str, int] = {}
    for char in fr:
        counts[char] = counts.get(char, 0) + 1
        if char in LIMITS and counts[char] > LIMITS[char]:
            return False
    if sum(char.isdigit() for char in fr) / len(fr) > 0.3:
        return False
    return True


def read_raw(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return f.readlines()


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.writelines(lines)


def text(line: str) -> str:
    return line[:-1] if line.endswith("\n") else line


def select_subset(
    indices: list[int], n: int, seed: int = 0
) -> list[int]:
    rng = random.Random(seed)
    return sorted(rng.sample(indices, n))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--en",
        type=Path,
        default=ROOT / "datasets/machine_translation/corpus_enfr_final.test.en",
    )
    parser.add_argument(
        "--fr",
        type=Path,
        default=ROOT / "datasets/machine_translation/corpus_enfr_final.test.fr",
    )
    parser.add_argument(
        "--fr-vocab",
        type=Path,
        default=ROOT / "models/nmt_fren_goldmaster_bpe.pt",
        help="vocabulary whose tgt BPE encoder defines the 120-token cutoff",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results/thesis_bleu/data",
    )
    parser.add_argument("--max-seq-len", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sizes", type=int, nargs="+", default=[5000, 20000])
    args = parser.parse_args()

    en_lines = read_raw(args.en)
    fr_lines = read_raw(args.fr)
    if len(en_lines) != len(fr_lines):
        raise SystemExit(
            f"line count mismatch: {len(en_lines)} en vs {len(fr_lines)} fr"
        )
    print(f"[info] loaded {len(en_lines)} pairs", flush=True)

    lowered_en = [line.lower() for line in en_lines]
    lowered_fr = [line.lower() for line in fr_lines]

    keep = [i for i, fr in enumerate(lowered_fr) if verify(fr)]
    dropped_verify = len(lowered_fr) - len(keep)
    print(
        f"[info] verify() dropped {dropped_verify}; {len(keep)} remain",
        flush=True,
    )

    vocab = torch.load(args.fr_vocab, map_location="cpu", weights_only=False)
    fr_bpe = BPE.from_dict(vocab["dict"]["tgt"])
    fr_bpe.mute()

    lengths = []
    for encoded in fr_bpe.transform([lowered_fr[i] for i in keep]):
        lengths.append(len(encoded))
    keep_bpe = [i for i, length in zip(keep, lengths) if length <= args.max_seq_len]
    dropped_bpe = len(keep) - len(keep_bpe)
    print(
        f"[info] BPE length filter dropped {dropped_bpe}; "
        f"{len(keep_bpe)} remain (total removed {len(en_lines) - len(keep_bpe)})",
        flush=True,
    )

    out_dir = args.out_dir
    full_en = [lowered_en[i] for i in keep_bpe]
    full_fr = [lowered_fr[i] for i in keep_bpe]
    write_lines(out_dir / "test_gm.en", full_en)
    write_lines(out_dir / "test_gm.fr", full_fr)
    print(f"[info] wrote {len(full_en)} lines to {out_dir}/test_gm.{{en,fr}}", flush=True)

    # Draw the largest sample, then derive smaller samples from it so that the
    # 5K decode is a strict subset of the 20K decode.
    largest = max(args.sizes)
    sampled_positions = select_subset(list(range(len(keep_bpe))), largest, args.seed)
    for size in sorted(args.sizes, reverse=True):
        if size == largest:
            positions = sampled_positions
        else:
            positions = select_subset(sampled_positions, size, args.seed)
        tag = f"{size // 1000}k"
        write_lines(out_dir / f"test_gm.{tag}.en", [full_en[i] for i in positions])
        write_lines(out_dir / f"test_gm.{tag}.fr", [full_fr[i] for i in positions])
        write_lines(
            out_dir / f"test_gm.{tag}.idx",
            [f"{i}\n" for i in positions],
        )
        print(
            f"[info] wrote {size} lines to {out_dir}/test_gm.{tag}.{{en,fr,idx}}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
