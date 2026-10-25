"""Helpers shared by the machine-translation training entry points."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
from torch.utils.data import DataLoader

from datasets import (
    Dataset,
    DatasetDict,
    Value,
    concatenate_datasets,
    load_dataset,
)
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
POLITICAL_DATASET = "thien/political"
PUBLICATIONS_DATASET = "thien/publications"

# name -> (Hub dataset, config, default French column)
STYLE_DATASETS = {
    "political": (POLITICAL_DATASET, "style", "text_fr"),
    "publications": (PUBLICATIONS_DATASET, None, "fr_text"),
}
# The plan's class order. ``class_encode_column`` sorts alphabetically, which
# would make democratic 0 and republican 1, so political is mapped explicitly.
POLITICAL_LABELS = {"republican": 0, "democratic": 1}

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
        f"Dataset has no split from {names!r}; available splits: {list(dataset.keys())}"
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


def _is_string_column(dataset: Dataset, column: str) -> bool:
    feature = dataset.features[column]
    return isinstance(feature, Value) and feature.dtype == "string"


def _encode_style_labels(dataset: Dataset, name: str) -> Dataset:
    """Turn string labels into integer classes."""
    if not _is_string_column(dataset, "label"):
        return dataset
    if name == "publications":
        return dataset.class_encode_column("label")

    unknown = sorted(set(dataset.unique("label")) - set(POLITICAL_LABELS))
    if unknown:
        raise ValueError(
            f"Unknown political labels {unknown}; expected one of "
            f"{sorted(POLITICAL_LABELS)}"
        )
    dataset = dataset.map(
        lambda batch: {"label": [POLITICAL_LABELS[label] for label in batch["label"]]},
        batched=True,
    )
    return dataset.cast_column("label", Value("int64"))


def _prepare_style_split(dataset: Dataset, name: str, source_column: str) -> Dataset:
    missing = {source_column, "text", "label"} - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"Style dataset {name!r} is missing column(s) {sorted(missing)}; "
            f"available columns: {dataset.column_names}"
        )
    dataset = dataset.rename_column(source_column, SOURCE).rename_column("text", TARGET)
    # Drop id, raw_text, filename, split, publication and any other leftovers.
    dataset = dataset.select_columns([SOURCE, TARGET, "label"])
    return _encode_style_labels(dataset, name)


def load_style_datasets(
    name: str,
    target_label: Optional[int] = None,
    num_proc: int = 8,
    path: Optional[Union[str, Path]] = None,
    source_column: Optional[str] = None,
    eval_path: Optional[Union[str, Path]] = None,
) -> Tuple[Dataset, Dataset]:
    """Load a labelled style corpus as ``(train, eval)``.

    ``name`` is ``"political"`` or ``"publications"``. The French column is
    renamed to SOURCE and ``text`` to TARGET; every other column is dropped, so
    only SOURCE, TARGET and ``label`` survive.

    ``source_column`` overrides the French column name. The Hub's political
    ``style`` config has no French column, so callers have to name the one they
    mean; the defaults are ``text_fr`` for political (the original local data)
    and ``fr_text`` for publications.

    ``path`` loads a local parquet instead of the Hub. A local file usually
    holds one split, which is returned as both train and eval. A file with an
    eval/validation/test split gives two different ones. ``eval_path`` loads a
    separate file for the evaluation split, so a local run can hold out data.
    """
    try:
        hub_name, config, default_source = STYLE_DATASETS[name]
    except KeyError:
        raise ValueError(
            f"Unknown style dataset {name!r}; expected one of {sorted(STYLE_DATASETS)}"
        ) from None
    source_column = source_column or default_source

    if path is not None:
        loaded = load_dataset("parquet", data_files=str(path))
        train = _prepare_style_split(
            _single_split(loaded, "train"), name, source_column
        )
        if eval_path is not None:
            loaded_eval = load_dataset("parquet", data_files=str(eval_path))
            eval_split = _prepare_style_split(
                _single_split(loaded_eval, "train"), name, source_column
            )
        else:
            try:
                eval_split = _single_split(
                    loaded, "eval", "validation", "valid", "test"
                )
            except ValueError:
                # A single-split local file is its own evaluation set.
                eval_split = train
            else:
                eval_split = _prepare_style_split(eval_split, name, source_column)
    else:
        loaded = load_dataset(hub_name, config, num_proc=num_proc, keep_in_memory=True)
        train = _prepare_style_split(
            _single_split(loaded, "train"), name, source_column
        )
        eval_split = _prepare_style_split(
            _single_split(loaded, "eval", "validation", "valid", "test"),
            name,
            source_column,
        )

    if target_label is not None:
        train = train.filter(lambda row: row["label"] == target_label)
        eval_split = eval_split.filter(lambda row: row["label"] == target_label)

    return train, eval_split


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
