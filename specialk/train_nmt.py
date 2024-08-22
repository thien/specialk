from __future__ import division

from pathlib import Path
from typing import Optional, Tuple, Union

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import AdvancedProfiler
from torch.utils.data import DataLoader

import datasets
from datasets import Dataset, concatenate_datasets, load_dataset
from specialk.core.constants import (
    LOGGING_DIR,
    LOGGING_PERF_NAME,
    PROJECT_DIR,
    RNN,
    SEED,
    SOURCE,
    TARGET,
    TRANSFORMER,
)
from specialk.core.utils import check_torch_device, log
from specialk.models.mt_model import RNNModule, TransformerModule
from specialk.models.tokenizer import (
    BPEVocabulary,
    SentencePieceVocabulary,
    Vocabulary,
    WordVocabulary,
)
from specialk.models.transformer.torch.pytorch_transformer import (
    PyTorchTransformerModule,
)

DEVICE: str = check_torch_device()

np.random.seed(SEED)  # if you're using numpy
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True)


def load_validation_dataset(src_lang: str, tgt_lang: str, num_proc=1) -> Dataset:
    ds = load_dataset(
        "wmt/wmt15", "fr-en", split=datasets.Split.VALIDATION, num_proc=num_proc
    )

    # Define a function to restructure the data
    def restructure(examples):
        return {
            SOURCE: [x[src_lang] for x in examples["translation"]],
            TARGET: [x[tgt_lang] for x in examples["translation"]],
        }

    # Apply the function to the dataset
    restructured_ds = ds.map(restructure, batched=True, remove_columns=ds.column_names)

    return restructured_ds


def load_training_dataset(src_lang: str, tgt_lang: str, num_proc=8) -> Dataset:
    """Datasets have been uploaded to huggingface."""

    dataset_sources = [
        "thien/gigatext",
        "thien/cc",
        "thien/un2k",
        "thien/europarl",
        "thien/news-commentary",
        "thien/globalvoices",
    ]
    ds: Dataset = concatenate_datasets(
        [
            load_dataset(d, num_proc=num_proc, keep_in_memory=True)["train"]
            for d in dataset_sources
        ]
    )
    ds = ds.shuffle(seed=SEED)
    ds = (
        ds.rename_column(src_lang, SOURCE)
        .rename_column(tgt_lang, TARGET)
        .remove_columns(["__index_level_0__"])
    )

    return ds


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
    """
    Prepare dataset into model interpretable data.

    In other words, performs tokenization, shuffling (if needed),
    and creates DataLoaders for the datasets.
    """
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
        tokenize, batched=True, num_proc=n_workers, cache_file_name=cache_path
    )
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        persistent_workers=persistent_workers,
        shuffle=shuffle,
        num_workers=n_workers,
    )
    return dataloader, (tokenizer, decoder_tokenizer)


def load_tokenizer(
    option: str = "word", max_length: int = 100
) -> Tuple[Union[BPEVocabulary, WordVocabulary, SentencePieceVocabulary], Path]:
    """Load tokenizer.

    Args:
        option (str, optional): _description_. Defaults to "word".

    Returns:
        Union[BPEVocabulary, WordVocabulary, SentencePieceVocabulary]: Returns tokenizer.
    """
    WORD, BPE, SPM = "word", "bpe", "spm"
    assert option in {WORD, BPE, SPM}

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
    elif option == SPM:
        tokenizer_filepath = dir_tokenizer / "sentencepiece" / "enfr.model"
        tokenizer = SentencePieceVocabulary.from_file(
            tokenizer_filepath, max_length=max_length
        )
    else:
        raise Exception("valid option not selected.")
    log.info("Loaded tokenizer", tokenizer=tokenizer, filepath=str(tokenizer_filepath))
    return tokenizer, tokenizer_filepath


def load_mono_tokenizer() -> Tuple[WordVocabulary, WordVocabulary, Path, Path]:
    dir_tokenizer = PROJECT_DIR / "assets" / "tokenizer"
    src_tokenizer_filepath: Path = dir_tokenizer / "de_small_word_moses"
    src_tokenizer = WordVocabulary.from_file(src_tokenizer_filepath)
    tgt_tokenizer_filepath: Path = dir_tokenizer / "en_small_word_moses"
    tgt_tokenizer = WordVocabulary.from_file(tgt_tokenizer_filepath)
    return src_tokenizer, tgt_tokenizer, src_tokenizer_filepath, tgt_tokenizer_filepath


def invert_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """This is for backtranslation."""
    df[SOURCE], df[TARGET] = df[TARGET], df[SOURCE]
    df["source_lang"], df["target_lang"] = df["target_lang"], df["source_lang"]
    return df


def main():
    PROD = DEVICE == "cuda"
    MODEL = TRANSFORMER
    TRANSFORMER_LEGACY = False
    DATASET_DIR = PROJECT_DIR / "datasets" / "machine_translation" / "parquets"
    # make cache dir
    CACHE_DIR = PROJECT_DIR / "cache"
    DATASET_CACHE_DIR = CACHE_DIR / "datasets"
    DATASET_CACHE_DIR.mkdir(exist_ok=True)
    N_WORKERS: int = 7
    if PROD:
        TASK_NAME = "nmt_model"
        # dataset configs
        PATH_VALID = Path("wmt15")
        PATH_TRAIN = Path("thien_mt_datasets")
        BACK_TRANSLATION = True

        BATCH_SIZE = 96
        MAX_SEQ_LEN = 100
        if MODEL == RNN:
            BATCH_SIZE = 192
            MAX_SEQ_LEN = 75
        N_EPOCHS = 6
        LOG_EVERY_N_STEPS = 20
        # model configs
        RNN_CONFIG = {
            "name": "lstm",
            "brnn": True,
            "learning_rate": 0.001,
            "accumulate_grad_batches": 1,
        }
        TRANSFORMER_CONFIG = {
            "name": "transformer",
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "n_heads": 8,
            "dim_model": 512,
            "n_warmup_steps": 4000,
            "learning_rate": 0.001,
            "accumulate_grad_batches": 10,  # multiply by batch_size to get eff. batch size
        }
        src_tokenizer, src_tokenizer_filepath = load_tokenizer("spm", MAX_SEQ_LEN)
        tgt_tokenizer, tgt_tokenizer_filepath = src_tokenizer, src_tokenizer_filepath

        TOKENIZER_CONFIG = {
            "vocabulary_size": src_tokenizer.vocab_size,
            "decoder_vocabulary_size": tgt_tokenizer.vocab_size,
            "sequence_length": src_tokenizer.max_length,
            "tokenizer": src_tokenizer,
            "decoder_tokenizer": tgt_tokenizer,
        }
    else:
        TASK_NAME = "nmt_model_dummy"
        PATH_VALID = DATASET_DIR / "de_en_30k_valid.parquet"
        PATH_TRAIN = DATASET_DIR / "de_en_30k_train.parquet"
        BACK_TRANSLATION = False
        BATCH_SIZE = 64 if DEVICE == "mps" else 32
        BATCH_SIZE = 192
        MAX_SEQ_LEN = 30
        src_tokenizer, tgt_tokenizer, src_tokenizer_filepath, tgt_tokenizer_filepath = (
            load_mono_tokenizer()
        )
        RNN_CONFIG = {
            "name": "lstm_smol",
            "rnn_size": 192,
            "d_word_vec": 128,
            "brnn": True,
            "learning_rate": 0.001,
            "accumulate_grad_batches": 1,
        }
        TRANSFORMER_CONFIG = {
            "name": "transformer_smol",
            "num_encoder_layers": 3,
            "num_decoder_layers": 3,
            "n_heads": 4,
            "dim_model": 128,
            "n_warmup_steps": 80,
            "learning_rate": 0.001,
            "accumulate_grad_batches": 4,
        }
        if TRANSFORMER_LEGACY:
            TRANSFORMER_CONFIG["name"] += "_legacy"
        N_EPOCHS = 60
        LOG_EVERY_N_STEPS = 20

    TOKENIZER_CONFIG = {
        "tokenizer": src_tokenizer,
        "decoder_tokenizer": tgt_tokenizer,
        "vocabulary_size": src_tokenizer.vocab_size,
        "decoder_vocabulary_size": tgt_tokenizer.vocab_size,
        "sequence_length": src_tokenizer.max_length,
    }
    RNN_CONFIG = {**RNN_CONFIG, **TOKENIZER_CONFIG}
    TRANSFORMER_CONFIG = {**TRANSFORMER_CONFIG, **TOKENIZER_CONFIG}

    # load dataset
    log.info("Loading datasets", train=str(PATH_TRAIN), valid=str(PATH_VALID))
    if not PROD:
        df_train = pd.read_parquet(PATH_TRAIN).sample(frac=1)  # shuffle.
        df_valid = pd.read_parquet(PATH_VALID)
        if BACK_TRANSLATION:
            log.info("Inverting columns to allow for backtranslation.")
            df_train = invert_df_columns(df_train)
            df_valid = invert_df_columns(df_valid)
            log.info("Backtranslation flag enabled.")
        else:
            log.info("Backtranslation is disabled.")

        train_dataset: Dataset = Dataset.from_pandas(df_train)
        valid_dataset: Dataset = Dataset.from_pandas(df_valid)
    else:
        train_dataset = load_training_dataset(
            src_lang="fr", tgt_lang="en", num_proc=N_WORKERS
        )
        valid_dataset = load_validation_dataset(
            src_lang="fr", tgt_lang="en", num_proc=N_WORKERS
        )
    log.info("Loaded dataset", train=train_dataset.shape, valid=valid_dataset.shape)

    # create dataloaders
    train_dataloader, _ = init_dataloader(
        train_dataset,
        src_tokenizer,
        BATCH_SIZE,
        decoder_tokenizer=tgt_tokenizer,
        shuffle=True,
        cache_path=DATASET_CACHE_DIR / PATH_TRAIN.name,
        n_workers=N_WORKERS,
    )
    val_dataloader, _ = init_dataloader(
        valid_dataset,
        src_tokenizer,
        BATCH_SIZE,
        decoder_tokenizer=tgt_tokenizer,
        shuffle=False,
        cache_path=DATASET_CACHE_DIR / PATH_VALID.name,
        n_workers=N_WORKERS,
    )
    log.info("Created dataset dataloaders.")

    if MODEL == RNN:
        task = RNNModule(**RNN_CONFIG)
        # is this needed?
        task.model.PAD = src_tokenizer.PAD
    else:
        if TRANSFORMER_LEGACY and not PROD:
            task = TransformerModule(**TRANSFORMER_CONFIG)
        else:
            task = PyTorchTransformerModule(**TRANSFORMER_CONFIG)

    log.info("model initialised.", model=task)

    hyperparams = {
        "batch_size": BATCH_SIZE,
        "src_tokenizer": src_tokenizer.__class__.__name__,
        "tgt_tokenizer": tgt_tokenizer.__class__.__name__,
        "dataset": "machine_translation",
        "src_tokenizer_path": src_tokenizer_filepath,
        "tgt_tokenizer_path": tgt_tokenizer_filepath,
        "max_sequence_length": src_tokenizer.max_length,
        "dataset_train_path": PATH_TRAIN,
        "dataset_valid_path": PATH_VALID,
    }
    log.info("Hyperparameters initialised.", hyperparams=hyperparams)

    logger = TensorBoardLogger(LOGGING_DIR, name=f"{TASK_NAME}/{task.name}")
    logger.log_hyperparams(params=hyperparams)

    checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor="val_loss")

    profiler = None
    if PROD:
        # profiling will slow prod down.
        profiler = AdvancedProfiler(dirpath=logger.log_dir, filename=LOGGING_PERF_NAME)

    REVIEW_RATE = len(train_dataloader) // 4  # for debugging purposes.
    if PROD:
        REVIEW_RATE = len(train_dataloader) // 70  # takes around 24 hours per epoch

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        accelerator=DEVICE,
        max_epochs=N_EPOCHS,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        logger=logger,
        profiler=profiler,
        precision=16,
        val_check_interval=REVIEW_RATE,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=task.kwargs["accumulate_grad_batches"],
        callbacks=[lr_monitor, checkpoint_callback],
    )

    trainer.fit(
        task, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    main()
