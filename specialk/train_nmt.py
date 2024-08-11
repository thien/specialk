from __future__ import division

from pathlib import Path
from typing import Optional, Tuple, Union

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import AdvancedProfiler
from torch.utils.data import DataLoader

from datasets import Dataset
from specialk.core.constants import (
    LOGGING_DIR,
    LOGGING_PERF_NAME,
    PROJECT_DIR,
    RNN,
    SOURCE,
    TARGET,
    TRANSFORMER,
)
from specialk.core.utils import check_torch_device, log
from specialk.models.mt_model import RNNModule
from specialk.models.tokenizer import (
    BPEVocabulary,
    SentencePieceVocabulary,
    Vocabulary,
    WordVocabulary,
)
from specialk.models.transformer.torch.pytorch_transformer import (
    PyTorchTransformerModule as TransformerModule,
)

DEVICE: str = check_torch_device()


def init_dataloader(
    dataset: Dataset,
    tokenizer: Vocabulary,
    batch_size: int,
    shuffle: bool,
    decoder_tokenizer: Optional[Vocabulary] = None,
    n_workers: int = 4,
    persistent_workers: bool = True,
    cache_path: Optional[Path] = None,
):
    if decoder_tokenizer is None:
        decoder_tokenizer = tokenizer

    n_map_workers = 0 if DEVICE == "cuda" else n_workers

    def tokenize(example):
        # perform tokenization at this stage.
        example[SOURCE] = tokenizer.to_tensor(example[SOURCE]).squeeze(0)
        example[TARGET] = decoder_tokenizer.to_tensor(example[TARGET]).squeeze(0)
        return example

    tokenized_dataset = dataset.with_format("torch").map(
        tokenize, batched=True, num_proc=n_map_workers, cache_file_name=str(cache_path)
    )
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        persistent_workers=persistent_workers,
        shuffle=shuffle,
        num_workers=n_workers,
    )
    return dataloader, tokenizer


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
    df[SOURCE], df[TARGET] = df[TARGET], df[SOURCE]
    df["source_lang"], df["target_lang"] = df["target_lang"], df["source_lang"]
    return df


def main():
    PROD = DEVICE == "cuda"
    PROD = True
    MODEL = TRANSFORMER
    DATASET_DIR = PROJECT_DIR / "datasets" / "machine_translation" / "parquets"
    # make cache dir
    CACHE_DIR = PROJECT_DIR / "cache"
    DATASET_CACHE_DIR = CACHE_DIR / "datasets"
    DATASET_CACHE_DIR.mkdir(exist_ok=True)
    if PROD:
        TASK_NAME = "nmt_model"
        # dataset configs
        PATH_VALID = DATASET_DIR / "corpus_enfr_final.val.parquet"
        PATH_TRAIN = DATASET_DIR / "corpus_enfr_final.train.parquet"
        BACK_TRANSLATION = True

        BATCH_SIZE = 64
        MAX_SEQ_LEN = 100
        if MODEL == RNN:
            BATCH_SIZE = 192
            MAX_SEQ_LEN = 75
        N_EPOCHS = 3
        LOG_EVERY_N_STEPS = 20
        # model configs
        RNN_CONFIG = {
            "name": "lstm",
            "brnn": True,
        }
        TRANSFORMER_CONFIG = {
            "name": "transformer",
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "n_heads": 8,
            "dim_model": 512,
            "n_warmup_steps": 4000,
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
        }
        TRANSFORMER_CONFIG = {
            "name": "transformer_smol",
            "num_encoder_layers": 3,
            "num_decoder_layers": 3,
            "n_heads": 4,
            "dim_model": 128,
            "n_warmup_steps": 80,
        }
        N_EPOCHS = 30
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
    df_train = pd.read_parquet(PATH_TRAIN).sample(frac=1)  # shuffle.
    df_valid = pd.read_parquet(PATH_VALID)
    log.info("Loaded dataset", df_train=df_train.shape, df_valid=df_valid.shape)

    if BACK_TRANSLATION:
        log.info("Inverting columns to allow for backtranslation.")
        df_train = invert_df_columns(df_train)
        df_valid = invert_df_columns(df_valid)
        log.info("Backtranslation flag enabled.")
    else:
        log.info("Backtranslation is disabled.")

    train_dataset: Dataset = Dataset.from_pandas(df_train)
    valid_dataset: Dataset = Dataset.from_pandas(df_valid)

    # create dataloaders
    train_dataloader, _ = init_dataloader(
        train_dataset,
        src_tokenizer,
        BATCH_SIZE,
        decoder_tokenizer=tgt_tokenizer,
        shuffle=True,
        cache_path=DATASET_CACHE_DIR / PATH_TRAIN.name,
    )
    val_dataloader, _ = init_dataloader(
        valid_dataset,
        src_tokenizer,
        BATCH_SIZE,
        decoder_tokenizer=tgt_tokenizer,
        shuffle=False,
        cache_path=DATASET_CACHE_DIR / PATH_VALID.name,
    )
    log.info("Created dataset dataloaders.")

    if MODEL == RNN:
        task = RNNModule(**RNN_CONFIG)
        # is this needed?
        task.model.PAD = src_tokenizer.PAD
    else:
        task = TransformerModule(**TRANSFORMER_CONFIG)
    if DEVICE == "cuda":
        # compile for gains.
        task = torch.compile(task)
    log.info("model initialized.", model=task)

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
        "learning_rate": task.configure_optimizers().defaults["lr"],
    }
    log.info("Showing hyperparameters.", hyperparams=hyperparams)

    logger = TensorBoardLogger(LOGGING_DIR, name=f"{TASK_NAME}/{task.name}")
    logger.log_hyperparams(params=hyperparams)
    profiler = None
    if PROD:
        profiler = AdvancedProfiler(dirpath=logger.log_dir, filename=LOGGING_PERF_NAME)
    trainer = pl.Trainer(
        accelerator=DEVICE,
        max_epochs=N_EPOCHS,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        logger=logger,
        profiler=profiler,
    )

    trainer.fit(
        task, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    main()
