from __future__ import division

from typing import Union

import lightning.pytorch as pl
import pandas as pd
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import AdvancedProfiler
from torch.utils.data import DataLoader

from datasets import Dataset
from specialk.core.constants import LOGGING_DIR, LOGGING_PERF_NAME, PROJECT_DIR
from specialk.core.utils import check_torch_device, log
from specialk.models.tokenizer import (
    BPEVocabulary,
    SentencePieceVocabulary,
    Vocabulary,
    WordVocabulary,
)
from specialk.models.transformer.pytorch_transformer import (
    PyTorchTransformerModule as TransformerModule,
)

DEVICE: str = check_torch_device()


def init_dataloader(
    dataset: Dataset, tokenizer: Vocabulary, batch_size: int, shuffle: bool
):
    def tokenize(example):
        # perform tokenization at this stage.
        example["source"] = tokenizer.to_tensor(example["source"]).squeeze(0)
        example["target"] = tokenizer.to_tensor(example["target"]).squeeze(0)
        return example

    tokenized_dataset = dataset.with_format("torch").map(tokenize, batched=True)
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        persistent_workers=True,
        shuffle=shuffle,
        num_workers=8,
    )
    return dataloader, tokenizer


def load_tokenizer(
    option: str = "word",
) -> Union[BPEVocabulary, WordVocabulary, SentencePieceVocabulary]:
    """_summary_

    Args:
        option (str, optional): _description_. Defaults to "word".

    Returns:
        Union[BPEVocabulary, WordVocabulary, SentencePieceVocabulary]: Returns tokenizer.
    """
    WORD, BPE, SPM = "word", "bpe", "spm"
    assert option in {WORD, BPE, SPM}

    dir_tokenizer = PROJECT_DIR / "assets" / "tokenizer"
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
            tokenizer_filepath, max_length=100
        )
    return tokenizer, tokenizer_filepath


def main():
    BATCH_SIZE = 64 if DEVICE == "mps" else 32

    # init tokenizer
    tokenizer, tokenizer_filepath = load_tokenizer("spm")
    log.info("Loaded tokenizer", tokenizer=tokenizer)

    # load dataset
    dataset_dir = PROJECT_DIR / "datasets" / "machine_translation" / "parquets"
    path_valid = dataset_dir / "corpus_enfr_final.val.parquet"
    path_train = dataset_dir / "corpus_enfr_final.train.parquet"

    train_dataset: Dataset = Dataset.from_pandas(
        pd.read_parquet(path_train).sample(n=100000, random_state=1)
    )
    valid_dataset: Dataset = Dataset.from_pandas(pd.read_parquet(path_valid))

    # create dataloaders
    train_dataloader, _ = init_dataloader(
        train_dataset, tokenizer, BATCH_SIZE, shuffle=True
    )
    val_dataloader, _ = init_dataloader(
        valid_dataset, tokenizer, BATCH_SIZE, shuffle=False
    )

    task = TransformerModule(
        name="transformer",
        vocabulary_size=tokenizer.vocab_size,
        sequence_length=tokenizer.max_length,
        tokenizer=tokenizer,
        num_encoder_layers=3,
        num_decoder_layers=3,
        n_heads=8,
        dim_model=512,
    )

    logger = TensorBoardLogger(LOGGING_DIR, name="nmt_model")
    logger.log_hyperparams(
        params={
            "batch_size": BATCH_SIZE,
            "tokenizer": tokenizer.__class__.__name__,
            "dataset": "machine_translation",
            "tokenizer_path": tokenizer_filepath,
        }
    )

    profiler = AdvancedProfiler(dirpath=logger.log_dir, filename=LOGGING_PERF_NAME)
    trainer = pl.Trainer(
        accelerator=DEVICE,
        max_epochs=5,
        log_every_n_steps=20,
        logger=logger,
        profiler=profiler,
    )

    trainer.fit(
        task, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    main()
