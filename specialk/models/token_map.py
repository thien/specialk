"""Fixed token maps between a seq2seq vocabulary and a classifier vocabulary.

A trained projection can satisfy the classifier without the decoder improving,
so the map uses surface forms and falls back to the classifier's unknown token.
"""

from __future__ import annotations

from typing import Any

import sentencepiece as spm
import torch
from torch import Tensor

# SentencePiece marks a word start with U+2581, GPT-2 style tokenizers use
# U+0120, and WordPiece marks a continuation with "##".
_WORD_START_MARKERS = ("\u2581", "\u0120")


def _normalise(token: str) -> str:
    """Strip tokenizer markers so equal surface forms compare equal."""
    for marker in _WORD_START_MARKERS:
        token = token.replace(marker, "")
    token = token.removeprefix("##")
    return token.strip()


def _sentencepiece(tokenizer: Any) -> spm.SentencePieceProcessor | None:
    inner = getattr(tokenizer, "vocab", None)
    return inner if isinstance(inner, spm.SentencePieceProcessor) else None


def _token_strings(tokenizer: Any, size: int) -> list[str] | None:
    """Return the token strings for ids ``0..size-1``, or ``None``."""
    if tokenizer is None:
        return None

    processor = _sentencepiece(tokenizer)
    if processor is not None:
        return [processor.IdToPiece(i) for i in range(size)]

    # HuggingFaceVocabulary keeps the real tokenizer one level down.
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None:
        tokenizer = inner

    if hasattr(tokenizer, "convert_ids_to_tokens"):
        return [tokenizer.convert_ids_to_tokens(i) or "" for i in range(size)]

    # WordVocabulary stores an idx -> label dictionary.
    labels = getattr(getattr(tokenizer, "vocab", None), "idxToLabel", None)
    if labels is not None:
        return [labels.get(i, "") for i in range(size)]

    return None


def _special_ids(tokenizer: Any) -> dict[str, int]:
    """Special token ids under our wrapper names and HuggingFace's names."""
    if tokenizer is None:
        return {}
    names = {
        "pad": ("PAD", "pad_token_id"),
        "unk": ("UNK", "unk_token_id"),
        "bos": ("BOS", "SOS", "bos_token_id"),
        "eos": ("EOS", "eos_token_id"),
        "cls": ("CLS", "cls_token_id"),
        "sep": ("SEP", "sep_token_id"),
    }
    found: dict[str, int] = {}
    for kind, attributes in names.items():
        for attribute in attributes:
            value = getattr(tokenizer, attribute, None)
            if value is not None:
                found[kind] = int(value)
                break
    return found


def _lookup(clf_tokens: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    """Case-sensitive and case-folded lookups from surface form to id."""
    exact: dict[str, int] = {}
    folded: dict[str, int] = {}
    for index, token in enumerate(clf_tokens):
        key = _normalise(token)
        if not key:
            continue
        exact.setdefault(key, index)
        folded.setdefault(key.lower(), index)
    return exact, folded


def build_token_map(
    seq_tokenizer: Any,
    clf_tokenizer: Any,
    seq_vocab_size: int,
    clf_vocab_size: int,
) -> tuple[Tensor, float]:
    """Map every seq2seq token id to one classifier token id.

    Returns the map and the share of ordinary seq2seq tokens that matched a
    classifier token by surface form. Everything else points at the
    classifier's unknown token.

    Raises:
        TypeError: when either tokenizer cannot list its tokens.
    """
    seq_tokens = _token_strings(seq_tokenizer, seq_vocab_size)
    clf_tokens = _token_strings(clf_tokenizer, clf_vocab_size)
    if seq_tokens is None:
        raise TypeError(
            f"Cannot list tokens for the seq2seq tokenizer "
            f"({type(seq_tokenizer).__name__})"
        )
    if clf_tokens is None:
        raise TypeError(
            f"Cannot list tokens for the classifier tokenizer "
            f"({type(clf_tokenizer).__name__})"
        )

    exact, folded = _lookup(clf_tokens)
    seq_specials = _special_ids(seq_tokenizer)
    clf_specials = _special_ids(clf_tokenizer)

    unk = clf_specials.get("unk", 0)
    ids = torch.full((seq_vocab_size,), unk, dtype=torch.long)

    pairs = (
        ("pad", "pad"),
        ("unk", "unk"),
        ("bos", "cls"),
        ("eos", "sep"),
    )
    for source, target in pairs:
        source_id = seq_specials.get(source)
        if source_id is None or source_id >= seq_vocab_size:
            continue
        target_id = clf_specials.get(target, clf_specials.get(source, unk))
        ids[source_id] = target_id

    special_ids = set(seq_specials.values())
    matched = 0
    ordinary = 0
    for index, token in enumerate(seq_tokens):
        if index >= seq_vocab_size or index in special_ids:
            continue
        ordinary += 1
        key = _normalise(token)
        if not key:
            continue
        clf_id = exact.get(key)
        if clf_id is None:
            clf_id = folded.get(key.lower())
        if clf_id is not None:
            ids[index] = clf_id
            matched += 1

    coverage = matched / ordinary if ordinary else 0.0
    return ids, coverage
