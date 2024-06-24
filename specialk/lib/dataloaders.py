from typing import Tuple, Optional, Union
from torch.utils.data import DataLoader
from specialk.core.dataset import TranslationDataset, paired_collate_fn
from specialk.core.utils import log


def init_classification_dataloaders(
    data: dict, batch_size: int, n_workers: int = 8
) -> Tuple[DataLoader, DataLoader]:
    """Initialise CNN Classifier DataLoaders for dataset.

    Note that onmt has their own dataset loader, but no need to use that if we could
    leverage existing PyTorch Dataloaders.

    Args:
        data (dict): object container containing dataset.
        batch_size (int): batch size for dataset iteration.
        n_workers (int, Optional): number of workers to operate on the dataloader.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation dataloaders.
    """
    src_word2idx = data["dicts"]["src"]  # it's the same as the tgt.

    DATASET_IS_BPE = "byte_pairs" in src_word2idx.keys()
    if DATASET_IS_BPE:
        log.info("BPE Tokenised input detected.")
        # we have BPE loaded (detection is a heuristic).
        src_byte_pairs = {x + "_": y for x, y in src_word2idx["byte_pairs"].items()}
        src_word2idx = {**src_byte_pairs, **src_word2idx["words"]}
    else:
        log.info("Space-Separated Tokenised input detected.")

    train_loader = DataLoader(
        TranslationDataset(
            src_word2idx=src_word2idx,
            tgt_word2idx=src_word2idx,
            src_insts=data["train"]["src"],
            tgt_insts=data["train"]["tgt"],
        ),
        num_workers=n_workers,
        batch_size=batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True,
    )

    valid_loader = DataLoader(
        TranslationDataset(
            src_word2idx=src_word2idx,
            tgt_word2idx=src_word2idx,
            src_insts=data["valid"]["src"],
            tgt_insts=data["valid"]["tgt"],
        ),
        num_workers=n_workers,
        batch_size=batch_size,
        collate_fn=paired_collate_fn,
    )

    # printing for sanity checks.
    show_first_n: int = 2
    log.debug(
        f"Showing first {show_first_n} rows of training dataset.",
        source=data["train"]["src"][:show_first_n],
        target=data["train"]["tgt"][:show_first_n],
    )
    log.debug(
        f"Showing first {show_first_n} rows of validation dataset.",
        source=data["valid"]["src"][:show_first_n],
        target=data["valid"]["tgt"][:show_first_n],
    )

    return train_loader, valid_loader
