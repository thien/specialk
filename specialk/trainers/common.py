"""Helpers shared by the machine-translation training entry points.

These used to be duplicated in ``train_nmt.py`` and ``train_model_decoder.py``.
Keeping them in one place means a dataset or tokenizer fix only has to be made
once.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader

from specialk.core.constants import PROJECT_DIR, SEED, SOURCE, TARGET
from specialk.core.utils import log
from specialk.models.tokenizer import (
    BPEVocabulary,
    SentencePieceVocabulary,
    Vocabulary,
    WordVocabulary,
)

WORD = "word"
BPE = "bpe"
SPM = "spm"
TOKENIZER_OPTIONS = (WORD, BPE, SPM)

MT_DATASET_SOURCES = (
    "thien/gigatext",
    "thien/cc",
    "thien/un2k",
    "thien/europarl",
    "thien/news-commentary",
    "thien/globalvoices",
)
MT_VALIDATION_DATASET = "wmt/wmt15"
PUBLICATIONS_DATASET = "thien/publications"

DATASET_DIR = PROJECT_DIR / "datasets" / "machine_translation" / "parquets"
CACHE_DIR = PROJECT_DIR / "cache" / "datasets"


def _strip_index_column(dataset: Dataset) -> Dataset:
    """Drop the pandas index column when the dataset was built from a CSV."""
    if "__index_level_0__" in dataset.column_names:
        return dataset.remove_columns(["__index_level_0__"])
    return dataset


def _single_split(dataset: Union[Dataset, DatasetDict], *names: str) -> Dataset:
    """Return one split, accepting a plain dataset as-is."""
    if isinstance(dataset, Dataset):
        return dataset
    for name in names:
        if name in dataset:
            return dataset[name]
    raise ValueError(
        f"Dataset has no split from {names!r}; available splits: "
        f"{list(dataset.keys())}"
    )


def load_mt_training_dataset(
    src_lang: str = "fr", tgt_lang: str = "en", num_proc: int = 8
) -> Dataset:
    """Load the concatenated EN-FR machine-translation corpus.

    The datasets have been uploaded to HuggingFace.
    """
    ds = concatenate_datasets(
        [
            _single_split(
                load_dataset(name, num_proc=num_proc, keep_in_memory=True), "train"
            )
            for name in MT_DATASET_SOURCES
        ]
    )
    ds = ds.shuffle(seed=SEED)
    ds = ds.rename_column(src_lang, SOURCE).rename_column(tgt_lang, TARGET)
    return _strip_index_column(ds)


def load_mt_validation_dataset(
    src_lang: str = "fr", tgt_lang: str = "en", num_proc: int = 1
) -> Dataset:
    """Load the WMT15 validation split."""
    ds = load_dataset(
        MT_VALIDATION_DATASET,
        "fr-en",
        split="validation",
        num_proc=num_proc,
    )

    def restructure(examples):
        return {
            SOURCE: [x[src_lang] for x in examples["translation"]],
            TARGET: [x[tgt_lang] for x in examples["translation"]],
        }

    return ds.map(restructure, batched=True, remove_columns=ds.column_names)


def load_publications_datasets(num_proc: int = 8) -> Tuple[Dataset, Dataset]:
    """Load the French-to-English publications corpus as (train, validation)."""
    ds = load_dataset(PUBLICATIONS_DATASET, num_proc=num_proc, keep_in_memory=True)

    def restructure(split: Dataset) -> Dataset:
        split = split.rename_column("fr_text", SOURCE).rename_column("text", TARGET)
        return _strip_index_column(split)

    return (
        restructure(_single_split(ds, "train")),
        restructure(_single_split(ds, "eval", "validation", "valid", "test")),
    )


def cache_filename(
    dataset_path: Union[str, Path],
    tokenizer: str,
    max_seq_len: int,
    back_translation: bool = False,
) -> str:
    """Name the token cache after every setting that changes the tokens.

    ``Dataset.map`` reuses a cache file without checking what produced it, so a
    stale key silently trains on the wrong tokens.
    """
    path = str(dataset_path)
    suffix = "_bt" if back_translation else ""
    # The digest keeps same-named files from different --train-path values apart.
    digest = hashlib.md5(path.encode()).hexdigest()[:8]
    return f"{Path(path).stem}_{digest}_{tokenizer}_{max_seq_len}{suffix}"


def init_dataloader(
    dataset: Dataset,
    tokenizer: Vocabulary,
    batch_size: int,
    shuffle: bool,
    decoder_tokenizer: Optional[Vocabulary] = None,
    n_workers: int = 8,
    persistent_workers: bool = True,
    cache_path: Optional[Union[Path, str]] = None,
) -> Tuple[DataLoader, Tuple[Vocabulary, Vocabulary]]:
    """Tokenize a dataset and wrap it in a ``DataLoader``."""
    if decoder_tokenizer is None:
        decoder_tokenizer = tokenizer

    def tokenize(example):
        example[SOURCE] = tokenizer.to_tensor(example[SOURCE]).squeeze(0)
        example[TARGET] = decoder_tokenizer.to_tensor(example[TARGET]).squeeze(0)
        return example

    if cache_path:
        if isinstance(cache_path, Path):
            cache_path = str(cache_path)
        if not cache_path.lower().endswith(".parquet"):
            cache_path = f"{cache_path}.parquet"

    tokenized_dataset = dataset.with_format("torch").map(
        tokenize, batched=True, num_proc=n_workers or None, cache_file_name=cache_path
    )
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        # DataLoader rejects persistent workers when workers are disabled.
        persistent_workers=persistent_workers and n_workers > 0,
        shuffle=shuffle,
        num_workers=n_workers,
    )
    return dataloader, (tokenizer, decoder_tokenizer)


def load_tokenizer(
    option: str = WORD, max_length: int = 100
) -> Tuple[Union[BPEVocabulary, WordVocabulary, SentencePieceVocabulary], Path]:
    """Load one of the bilingual FR-EN tokenizers."""
    if option not in TOKENIZER_OPTIONS:
        raise ValueError(
            f"Unknown tokenizer {option!r}; expected one of {TOKENIZER_OPTIONS}"
        )

    dir_tokenizer = PROJECT_DIR / "assets" / "tokenizer"
    tokenizer_filepath: Path = dir_tokenizer
    tokenizer = None
    if option == BPE:
        tokenizer_filepath = dir_tokenizer / "fr_en_bpe"
        tokenizer = BPEVocabulary.from_file(tokenizer_filepath)
        tokenizer.vocab._progress_bar = iter
    elif option == WORD:
        tokenizer_filepath = dir_tokenizer / "fr_en_word_moses"
        tokenizer = WordVocabulary.from_file(tokenizer_filepath)
    else:
        tokenizer_filepath = dir_tokenizer / "sentencepiece" / "enfr.model"
        tokenizer = SentencePieceVocabulary.from_file(
            tokenizer_filepath, max_length=max_length
        )
    log.info("Loaded tokenizer", tokenizer=tokenizer, filepath=str(tokenizer_filepath))
    return tokenizer, tokenizer_filepath


def load_mono_tokenizer() -> Tuple[WordVocabulary, WordVocabulary, Path, Path]:
    """Load the small German/English tokenizers used for local smoke runs."""
    dir_tokenizer = PROJECT_DIR / "assets" / "tokenizer"
    src_tokenizer_filepath: Path = dir_tokenizer / "de_small_word_moses"
    src_tokenizer = WordVocabulary.from_file(src_tokenizer_filepath)
    tgt_tokenizer_filepath: Path = dir_tokenizer / "en_small_word_moses"
    tgt_tokenizer = WordVocabulary.from_file(tgt_tokenizer_filepath)
    return (
        src_tokenizer,
        tgt_tokenizer,
        src_tokenizer_filepath,
        tgt_tokenizer_filepath,
    )


def invert_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Swap source/target columns; used for back-translation."""
    df[SOURCE], df[TARGET] = df[TARGET], df[SOURCE]
    df["source_lang"], df["target_lang"] = df["target_lang"], df["source_lang"]
    return df
