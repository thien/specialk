import pandas as pd
from specialk.models.utils.lid import FastTextLID
from specialk.core import log
import numpy as np
from typing import Optional, Tuple, Any, Iterable
from tqdm import tqdm
from specialk.core.sanitisation import (
    fix_unicode,
    can_parse_html_pattern,
    is_valid_text,
)
from structlog.contextvars import (
    bound_contextvars,
)
import warnings
from specialk.core.parallel import ParallelProcessor

tqdm.pandas()
CLEAN = "clean"
REASON = "reason"


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
        """Init DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with columns that include src_col, tgt_col.
            src_col (str): column name corresponding to source text (str).
            tgt_col (str): column name corresponding to target text (str).
        """
        self.df = df
        self.src_col = src_col
        self.tgt_col = tgt_col
        self._lid = FastTextLID()

        # we'll add additional columns
        self.df[CLEAN] = True
        self.df[REASON] = None

    def _get_valid_rows(self) -> pd.DataFrame:
        """Return a boolean mask for valid rows. This is so we can
        only perform parallel computation on the valid entities."""
        return self.df[CLEAN]

    def normalise(self):
        """Perform unicode normalization on DataFrame."""
        with bound_contextvars(filter="unicode fix and norm"):
            log.info("Running normalization on rows.")

            for column in [self.src_col, self.tgt_col]:
                self._filter_valid_html_parseable(column)

            valid_rows = self._get_valid_rows()

            # TODO: Pandas screams when I do this because it'll get
            # deprecated in 3.0, but when I do self.df.loc[valid_rows, self.src_col]
            # the kernel dies. Need to integrate chunk_allocate_on_df().
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                warnings.simplefilter(
                    action="ignore", category=pd.errors.SettingWithCopyWarning
                )
                for column in [self.src_col, self.tgt_col]:
                    self.df[column].loc[valid_rows] = ParallelProcessor.process(
                        fix_unicode, self.df.loc[valid_rows, column]
                    )

            log.info("Finished running normalization on rows.")

    def filter_sanity_check(self):
        """Perform general heuristic based filtering on the text."""
        if len(self.df) < 1:
            return
        log.info("Running sanity check")
        with bound_contextvars(filter="sanity_check"):
            for column in [self.src_col, self.tgt_col]:
                self._filter_sanity_check_on_column(column)
        log.info("Finished running sanity checks.")

    def _filter_valid_html_parseable(self, column: str) -> None:
        """Perform is_valid_text on given column."""
        rows = self._get_valid_rows()
        texts = self.df.loc[rows, column]
        texts_valid = np.array(ParallelProcessor.process(can_parse_html_pattern, texts))

        valid_series = pd.Series(texts_valid, index=rows[rows].index)
        invalid_series = valid_series[~valid_series]  # only want the negative values.

        self.df.loc[invalid_series.index, CLEAN] = False
        self.df.loc[invalid_series.index, REASON] = f"{column} cannot parse HTML"

        n_total, n_valid = texts_valid.size, texts_valid.sum()
        n_dropped = n_total - n_valid

        log.info(
            f"calculated {((n_valid/n_total)*100):.2f}% {column} rows valid",
            total=n_total,
            valid=n_valid,
            dropped=n_dropped,
        )

    def _filter_sanity_check_on_column(self, column: str) -> None:
        """Perform is_valid_text on given column."""
        rows = self._get_valid_rows()
        texts = self.df.loc[rows, column]
        texts_valid = np.array(ParallelProcessor.process(is_valid_text, texts))

        valid_series = pd.Series(texts_valid, index=rows[rows].index)
        invalid_series = valid_series[~valid_series]  # only want the negative values.

        self.df.loc[invalid_series.index, CLEAN] = False
        self.df.loc[invalid_series.index, REASON] = f"{column} fails sanity check"

        n_total, n_valid = texts_valid.size, texts_valid.sum()
        n_dropped = n_total - n_valid

        log.info(
            f"calculated {((n_valid/n_total)*100):.2f}% {column} rows valid",
            total=n_total,
            valid=n_valid,
            dropped=n_dropped,
        )

    def filter_lang_id(
        self,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        prob_threshold: float = 0.9,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter dataset by Lang ID."""
        with bound_contextvars(filter="lang_id", threshold=prob_threshold):
            for column, language in [
                (self.src_col, src_lang),
                (self.tgt_col, tgt_lang),
            ]:
                self._filter_lang_id(column, prob_threshold, language)
            log.info("Finished running Language ID on rows.")

    def _filter_lang_id(
        self, column: str, threshold: float, lang: Optional[str] = None
    ) -> None:
        rows = self._get_valid_rows()

        labels, scores = self._lid.lid_scores(self.df.loc[rows, column])

        if lang is None:
            log.info(f"{column} language is being approximated.")
            lang = most_frequent_item_numpy(labels[0])
            log.info(f"Approximated src_labels to be {lang}.")

        high_conf_row = scores[:, 0] >= threshold
        row_label_match = labels[:, 0] == lang

        high_conf_label = high_conf_row & row_label_match

        valid_series = pd.Series(high_conf_label, index=rows[rows].index)
        invalid_series = valid_series[~valid_series]

        self.df.loc[invalid_series.index, CLEAN] = False
        self.df.loc[invalid_series.index, REASON] = (
            f"{column} low confidence (<{threshold}) for {lang} on lang_id"
        )

        n_total, n_valid = high_conf_label.size, high_conf_label.sum()
        n_dropped = n_total - n_valid

        log.info(
            f"calculated {((n_valid/n_total)*100):.2f}% {column} rows valid",
            total=n_total,
            valid=n_valid,
            dropped=n_dropped,
        )
