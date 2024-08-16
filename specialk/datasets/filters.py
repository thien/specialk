import pandas as pd
from specialk.models.utils.lid import FastTextLID
from specialk.core import log
import numpy as np
from typing import Optional, Tuple, Any, Iterable
from tqdm import tqdm
from specialk.core.sanitisation import fix_unicode
from structlog.contextvars import (
    bound_contextvars,
)
from specialk.core.parallel import ParallelProcessor

tqdm.pandas()
CLEAN = "clean"
REASON = "reason"

lims = {
    ".": 8,
    "(": 8,
    ")": 8,
    ",": 8,
    "@": 8,
}


def is_valid_text(x: str) -> bool:
    """
    Heuristic based text validity, gopher style.

    (yeah, heuristic based filtering has been
    around for ages, I know).
    """
    if len(x.strip()) < 1:
        return False

    # if the text contains a lot of numbers,
    if sum(c.isdigit() for c in x) / len(x) > 0.4:
        return False

    # check ratio of characters.
    base = {}
    for char in x:
        if char not in base:
            base[char] = 0
        base[char] += 1
        if char in lims:
            if lims[char] < base[char]:
                return False
    return True


def most_frequent_item_numpy(arr: np.array) -> Any:
    """Return the most frequent in the array."""
    unique, counts = np.unique(arr, return_counts=True)
    return unique[np.argmax(counts)]


def chunk_allocate_on_df(
    df: pd.DataFrame,
    column: str,
    row_indexer: pd.Series,
    values: Iterable[Any],
    chunk_size: int = 100000,
):
    for start in tqdm(range(0, len(df), chunk_size), desc="chunk storing"):
        end = start + chunk_size
        chunk_valid_rows = row_indexer[start:end]
        chunk_cleaned_rows = values[start:end]
        df.loc[chunk_valid_rows, column] = chunk_cleaned_rows


class PreTrainingFilter:
    """Package the filtering into a single class. This is mostly for my sanity."""

    def __init__(self, df: pd.DataFrame, src_col: str, tgt_col: str):
        self.df = df
        self.src_col = src_col
        self.tgt_col = tgt_col
        self._lid = FastTextLID()

        # we'll add additional columns
        self.df[CLEAN] = True
        self.df[REASON] = None

    def _get_valid_rows(self):
        """Return a boolean mask for valid rows."""
        return self.df[CLEAN]

    def normalise(self):
        with bound_contextvars(filter="unicode fix and norm"):
            log.info("Running normalization on rows.")
            valid_rows = self._get_valid_rows()
            self.df[self.src_col].loc[valid_rows] = ParallelProcessor.process(
                fix_unicode, self.df.loc[valid_rows, self.src_col]
            )

            self.df[self.tgt_col].loc[valid_rows] = ParallelProcessor.process(
                fix_unicode, self.df.loc[valid_rows, self.tgt_col]
            )

            log.info("Finished running normalization on rows.")

    def filter_sanity_check(self):
        if len(self.df) < 1:
            return
        log.info("Running sanity check")
        with bound_contextvars(filter="sanity_check"):
            # get the rows we want to perform operations on.
            valid_rows = self._get_valid_rows()
            src_texts = self.df.loc[valid_rows, self.src_col]
            src_valid = np.array(ParallelProcessor.process(is_valid_text, src_texts))

            self.df.loc[~src_valid, CLEAN] = False
            self.df.loc[~src_valid, REASON] = "src fails sanity check"

            log.info(
                f"calculated {(src_valid.sum()/src_valid.size):.2f} src rows valid",
                total=src_valid.size,
                valid=src_valid.sum(),
            )

            valid_rows = self._get_valid_rows()  # Update valid_rows after src check

            # logic to get valid texts.
            tgt_texts = self.df.loc[valid_rows, self.tgt_col]
            tgt_valid = np.array(ParallelProcessor.process(is_valid_text, tgt_texts))

            tgt_valid_series = pd.Series(tgt_valid, index=valid_rows[valid_rows].index)
            # only want the negative values.
            tgt_valid_series = tgt_valid_series[~tgt_valid_series]

            self.df.loc[tgt_valid_series.index, CLEAN] = False
            self.df.loc[tgt_valid_series.index, REASON] = "tgt fails sanity check"

            log.info(
                f"calculated {(tgt_valid.sum()/tgt_valid.size):.2f} tgt rows valid",
                total=tgt_valid.size,
                valid=tgt_valid.sum(),
            )
            log.info("Finished running sanity check on rows.")
        log.info("Finished running sanity checks.")

    def filter_lang_id(
        self,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        prob_threshold: float = 0.9,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter dataset by Lang ID."""
        with bound_contextvars(filter="lang_id", threshold=prob_threshold):
            log.info("Running Language ID on rows.")
            valid_rows = self._get_valid_rows()

            src_labels, src_scores = self._lid.lid_scores(
                self.df.loc[valid_rows, self.src_col]
            )

            if src_lang is None:
                log.info("src_lang is being approximated.")
                src_lang = most_frequent_item_numpy(src_labels[0])
                log.info(f"Approximated src_labels to be {src_lang}.")

            high_conf_row = src_scores[:, 0] >= prob_threshold
            row_label_match = src_labels[:, 0] == src_lang

            high_conf_label = high_conf_row & row_label_match

            self.df.loc[valid_rows[valid_rows].index[~high_conf_label], CLEAN] = False
            self.df.loc[valid_rows[valid_rows].index[~high_conf_label], REASON] = (
                "src low confidence of src_language on lid"
            )

            valid_rows = self._get_valid_rows()  # Update valid_rows after src check
            tgt_labels, tgt_scores = self._lid.lid_scores(
                self.df.loc[valid_rows, self.tgt_col]
            )

            if tgt_lang is None:
                log.info("tgt_lang is being approximated.")
                tgt_lang = most_frequent_item_numpy(tgt_labels[0])
                log.info(f"Approximated tgt_labels to be {tgt_lang}.")

            high_conf_row = tgt_scores[:, 0] >= prob_threshold
            row_label_match = tgt_labels[:, 0] == tgt_lang

            high_conf_label = high_conf_row & row_label_match

            self.df.loc[valid_rows[valid_rows].index[~high_conf_label], CLEAN] = False
            self.df.loc[valid_rows[valid_rows].index[~high_conf_label], REASON] = (
                "tgt low confidence of tgt_language on lid"
            )
            log.info("Finished running Language ID on rows.")
