import pandas as pd
import numpy as np
from typing import Optional, Any, Iterable
from tqdm import tqdm
from specialk.models.utils.lid import FastTextLID
from specialk.core import log
from specialk.core.sanitisation import (
    fix_unicode,
    can_parse_html_pattern,
    is_valid_text,
)
from structlog.contextvars import bound_contextvars
import warnings
from specialk.core.parallel import ParallelProcessor

tqdm.pandas()
CLEAN = "clean"
REASON = "reason"


def most_frequent_item_numpy(arr: np.array) -> Any:
    """Return the most frequent item in the array."""
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
    """Package the filtering into a single class for one column."""

    def __init__(
        self,
        df: pd.DataFrame,
        col: str,
        lid: Optional[FastTextLID] = None,
        init_already=False,
    ):
        """Init DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with column that includes col.
            col (str): column name corresponding to text (str).
            lid (Optional[FastTextID]): LanguageID object. Initialises one if
                it hasn't already been done so.
            init_already (Optional[bool]): If set, skips creating a "clean" and
                "reason" column.
        """
        self.df = df
        self.col = col
        self._lid = lid if lid else FastTextLID()

        if not init_already:
            # we'll add additional columns
            self.df[CLEAN] = True
            self.df[REASON] = ""

    def _get_valid_rows(self) -> pd.DataFrame:
        """Return a boolean mask for valid rows."""
        return self.df[CLEAN]

    def normalise(self):
        """Perform unicode normalization on DataFrame."""
        with bound_contextvars(filter="unicode fix and norm"):
            log.info("Running normalization on rows.")

            self.filter_valid_html_parseable()

            valid_rows = self._get_valid_rows()

            # TODO: Pandas screams when I do this because it'll get
            # deprecated in 3.0, but when I do self.df.loc[valid_rows, self.src_col]
            # the kernel dies. Need to integrate chunk_allocate_on_df().
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                warnings.simplefilter(
                    action="ignore", category=pd.errors.SettingWithCopyWarning
                )
                self.df[self.col].loc[valid_rows] = ParallelProcessor.process(
                    fix_unicode, self.df.loc[valid_rows, self.col]
                )

            log.info("Finished running normalization on rows.")

    def filter_sanity_check(self):
        """Perform general heuristic based filtering on the text."""
        if len(self.df) < 1:
            return
        log.info("Running sanity check")
        with bound_contextvars(filter="sanity_check"):
            rows = self._get_valid_rows()
            texts_valid = np.array(
                ParallelProcessor.process(is_valid_text, self.df.loc[rows, self.col])
            )

            valid_series = pd.Series(texts_valid, index=rows[rows].index)
            invalid_series = valid_series[~valid_series]

            self.df.loc[invalid_series.index, CLEAN] = False
            self.df.loc[invalid_series.index, REASON] = f"{self.col} fails sanity check"

            self.log_difference(texts_valid, self.col)

        log.info("Finished running sanity checks.")

    def filter_valid_html_parseable(self) -> None:
        """Perform is_valid_text on given column."""
        rows = self._get_valid_rows()
        texts_valid = np.array(
            ParallelProcessor.process(
                can_parse_html_pattern, self.df.loc[rows, self.col]
            )
        )

        valid_series = pd.Series(texts_valid, index=rows[rows].index)
        invalid_series = valid_series[~valid_series]

        self.df.loc[invalid_series.index, CLEAN] = False
        self.df.loc[invalid_series.index, REASON] = f"{self.col} cannot parse HTML"

        self.log_difference(texts_valid, self.col)

    def filter_lang_id(
        self,
        lang: Optional[str] = None,
        prob_threshold: float = 0.9,
    ) -> None:
        """Filter dataset by Lang ID."""
        with bound_contextvars(filter="lang_id", threshold=prob_threshold):
            rows = self._get_valid_rows()

            labels, scores = self._lid.lid_scores(self.df.loc[rows, self.col])

            if lang is None:
                log.info(f"{self.col} language is being approximated.")
                lang = most_frequent_item_numpy(labels[0])
                log.info(f"Approximated labels to be {lang}.")

            high_conf_row = scores[:, 0] >= prob_threshold
            row_label_match = labels[:, 0] == lang

            high_conf_label = high_conf_row & row_label_match

            valid_series = pd.Series(high_conf_label, index=rows[rows].index)
            invalid_series = valid_series[~valid_series]

            self.df.loc[invalid_series.index, CLEAN] = False
            self.df.loc[invalid_series.index, REASON] = (
                f"{self.col} low confidence (<{prob_threshold}) for {lang} on lang_id"
            )

            self.log_difference(high_conf_label, self.col)

            log.info("Finished running Language ID on rows.")

    @staticmethod
    def log_difference(results: np.array, column: str):
        """Pretty render log difference in changes."""
        n_total, n_valid = results.size, results.sum()
        n_dropped = n_total - n_valid

        log.info(
            f"calculated {((n_valid/n_total)*100):.2f}% {column} rows valid",
            total=n_total,
            valid=n_valid,
            dropped=n_dropped,
        )


class ParallelPreTrainingFilter(PreTrainingFilter):
    """Package the filtering into a single class for parallel corpus."""

    def __init__(self, df: pd.DataFrame, src_col: str, tgt_col: str):
        """Init DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with columns that include src_col, tgt_col.
            src_col (str): column name corresponding to source text (str).
            tgt_col (str): column name corresponding to target text (str).
        """
        super().__init__(df, src_col)
        self.src_col = src_col
        self.tgt_col = tgt_col
        self.src_filter = PreTrainingFilter(df, src_col, self._lid, init_already=True)
        self.tgt_filter = PreTrainingFilter(df, tgt_col, self._lid, init_already=True)

    def normalise(self):
        """Perform unicode normalization on DataFrame for both columns."""
        self.src_filter.normalise()
        self.tgt_filter.normalise()
        self._sync_clean_and_reason()

    def filter_sanity_check(self):
        """Perform general heuristic based filtering on both columns."""
        self.src_filter.filter_sanity_check()
        self.tgt_filter.filter_sanity_check()
        self._sync_clean_and_reason()

    def filter_lang_id(
        self,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        prob_threshold: float = 0.9,
    ) -> None:
        """Filter dataset by Lang ID for both columns."""
        self.src_filter.filter_lang_id(src_lang, prob_threshold)
        self.tgt_filter.filter_lang_id(tgt_lang, prob_threshold)
        self._sync_clean_and_reason()

    def _sync_clean_and_reason(self):
        """Synchronize CLEAN and REASON columns between source and target filters."""
        self.df[CLEAN] = self.src_filter.df[CLEAN] & self.tgt_filter.df[CLEAN]
        self.df[REASON] = self.src_filter.df[REASON].fillna(self.tgt_filter.df[REASON])
