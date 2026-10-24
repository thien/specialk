#!/usr/bin/env python
"""Beam-decode a text file with the thesis-era Transformer checkpoints.

This mirrors the old ``TransformerModel.load_testdata`` / ``translate`` path:
lowercase, BPE-encode, reclip to the model's maximum length, add <s>/</s>,
run the OpenNMT-style beam search from ``specialk.models.transformer.legacy``,
then merge BPE and clip at </s>.

Output is one detokenized-by-BPE (but still Moses-tokenized) line per source
line.  The file is flushed as it goes; on restart, completed lines are skipped.
"""

import argparse
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import specialk.core.constants as Constants  # noqa: E402
from specialk.core.dataset import TranslationDataset, collate_fn  # noqa: E402
from specialk.models.tokenizers.bpe import Encoder as BPE  # noqa: E402
from specialk.models.transformer.legacy.Models import Transformer  # noqa: E402
from specialk.models.transformer.legacy.Translator import Translator  # noqa: E402


def strip_module(state_dict: dict) -> "OrderedDict[str, torch.Tensor]":
    if state_dict and next(iter(state_dict)).startswith("module."):
        return OrderedDict((key[7:], value) for key, value in state_dict.items())
    return OrderedDict(state_dict)


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def build_model(settings) -> Transformer:
    return Transformer(
        settings.src_vocab_size,
        settings.tgt_vocab_size,
        settings.max_token_seq_len,
        tgt_emb_prj_weight_sharing=settings.proj_share_weight,
        emb_src_tgt_weight_sharing=settings.embs_share_weight,
        d_k=settings.d_k,
        d_v=settings.d_v,
        d_model=settings.d_model,
        d_word_vec=settings.d_word_vec,
        d_inner=settings.d_inner_hid,
        n_layers=settings.layers,
        n_head=settings.n_head,
        dropout=settings.dropout,
    )


def load_model(encoder_path: Path, decoder_path: Path, device: torch.device):
    encoder = load_checkpoint(encoder_path)
    decoder = load_checkpoint(decoder_path)

    settings = encoder["settings"]
    model = build_model(settings)
    model.encoder.load_state_dict(strip_module(encoder["model"]))
    model.decoder.load_state_dict(strip_module(decoder["model"]))
    model.generator.load_state_dict(strip_module(decoder["generator"]))
    model.to(device)
    model.eval()
    return model, settings, encoder


def reclip(raw: str, encoded: list[int], bpe: BPE, max_len: int) -> list[int]:
    """Trim the raw sentence until its BPE encoding fits (old preprocess.reclip)."""
    if len(encoded) <= max_len:
        return encoded
    tokens = raw.split()
    subtract = 1
    while len(encoded) > max_len:
        reference = " ".join(tokens[:-subtract])
        subtract += 1
        encoded = next(iter(bpe.transform([reference])))
    return encoded


def read_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def read_indices(path: Path) -> list[int]:
    return [int(line) for line in read_lines(path) if line.strip()]


def translate_decode_bpe(hypotheses: list[list[list[int]]], bpe: BPE) -> list[str]:
    """Port of the old NMTModel.translate_decode_bpe."""
    first = [hypothesis[0] for hypothesis in hypotheses]
    lines = []
    for sequence in bpe.inverse_transform(first):
        tokens = sequence.split()
        index = 0
        for j, token in enumerate(tokens):
            if token == Constants.EOS_WORD:
                if j == 0:
                    continue
                index = j
                break
        line = " ".join(tokens[:index])
        if len(line.strip()) < 1:
            line = Constants.UNK_WORD
        lines.append(line)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True, type=Path)
    parser.add_argument("--decoder", required=True, type=Path)
    parser.add_argument("--vocab", required=True, type=Path)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--n-best", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--indices",
        type=Path,
        help="file with one 0-based source line number per line (in output order)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--device", choices=["mps", "cpu"], default=None
    )
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[info] device: {device}", flush=True)

    model, settings, encoder_chkpt = load_model(
        args.encoder, args.decoder, device
    )
    print(
        "[info] model: data={} epoch={} max_token_seq_len={} d_model={} n_head={} "
        "src_vocab={} tgt_vocab={}".format(
            getattr(settings, "data", None),
            encoder_chkpt.get("epoch"),
            settings.max_token_seq_len,
            settings.d_model,
            settings.n_head,
            settings.src_vocab_size,
            settings.tgt_vocab_size,
        ),
        flush=True,
    )

    vocab_data = torch.load(args.vocab, map_location="cpu", weights_only=False)
    vocab_settings = vocab_data["settings"]
    src_bpe = BPE.from_dict(vocab_data["dict"]["src"])
    tgt_bpe = BPE.from_dict(vocab_data["dict"]["tgt"])

    sources = read_lines(args.src)
    if args.indices:
        indices = read_indices(args.indices)
        sources = [sources[i] for i in indices]
    if args.limit:
        sources = sources[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    if args.out.exists():
        with args.out.open(encoding="utf-8") as f:
            done = sum(1 for _ in f)
        print(f"[info] resuming after {done} completed lines", flush=True)

    case_sensitive = getattr(vocab_settings, "case_sensitive", False)
    max_len = vocab_settings.max_token_seq_len - 2
    sos, eos = Constants.SOS, Constants.EOS

    src_insts = []
    for raw in sources[done:]:
        text = raw if case_sensitive else raw.lower()
        encoded = next(iter(src_bpe.transform([text])))
        encoded = reclip(text.strip(), encoded, src_bpe, max_len)
        src_insts.append([sos] + encoded + [eos])

    if not src_insts:
        print("[info] nothing to do", flush=True)
        return 0

    # BPE ids for the model are the ids stored in the vocabulary; the old
    # TranslationDataset only used the dictionaries for sizing/introspection.
    src_word2idx = {**src_bpe.word_vocab, **src_bpe.bpe_vocab}
    tgt_word2idx = {**tgt_bpe.word_vocab, **tgt_bpe.bpe_vocab}
    loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=src_word2idx,
            tgt_word2idx=tgt_word2idx,
            src_insts=src_insts,
        ),
        num_workers=0,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )

    # Translator's constructor reads legacy CLI attributes even when the model
    # is supplied directly (new=False).  Give it what it expects, then point it
    # at the device we actually loaded the model on.
    args.cuda_device = None
    args.cuda = False
    args.beam_size = args.beam
    translator = Translator(args, new=False)
    translator.device = device
    translator.model = model
    translator.max_token_seq_len = settings.max_token_seq_len

    started = time.time()
    written = 0
    with args.out.open("a", encoding="utf-8") as out:
        for batch in loader:
            hypotheses, _ = translator.translate_batch(*batch)
            for line in translate_decode_bpe(hypotheses, tgt_bpe):
                out.write(line + "\n")
                written += 1
            out.flush()
            elapsed = time.time() - started
            rate = written / elapsed if elapsed else 0
            print(
                f"[info] {done + written}/{len(sources)} lines "
                f"({rate:.2f} lines/s)",
                flush=True,
            )

    print(f"[info] wrote {written} lines to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
