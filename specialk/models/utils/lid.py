import multiprocessing as mp
from huggingface_hub import hf_hub_download
import fasttext
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Generator, Iterable, Hashable, Any
from enum import Enum


class LanguageCode(Enum):
    """This is taken outside of FastTextLID.languages so I can use it
    for type hinting."""

    @classmethod
    def from_string(cls, code: str):
        # Handle both formats: with and without "__label__" prefix
        clean_code = code.replace("__label__", "")
        for member in cls:
            if member.value == clean_code:
                return member
        raise ValueError(f"'{code}' is not a valid LanguageCode")


class FastTextLID:
    """FastText wrapper."""

    def __init__(
        self, repo_id="facebook/fasttext-language-identification", filename="model.bin"
    ):
        """Loads huggingface repo; downloads model asset. Initiates with this model."""
        self.model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        self.model = fasttext.load_model(self.model_path)
        self.pbar_label = "FastText LID"

    @property
    def language(self) -> Enum:
        """Generate Enum for {ISO 639-3}_{ISO 15924} code."""
        language_dict = {}
        for code in self.model.get_labels():
            # Strip the "__label__" prefix and split the remaining code
            clean_code = code.replace("__label__", "")
            lang, script = clean_code.split("_")
            enum_name = f"{lang.upper()}_{script}"
            enum_value = clean_code
            language_dict[enum_name] = enum_value

        return LanguageCode("LanguageCode", language_dict)


    def process_batch(
        self, batch: List[str], k=1, threshold=0.0, on_unicode_error="strict"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate FastText LID labels and probs.

        From the FastText lib:
            Given a string, get a list of labels and a list of
            corresponding probabilities. k controls the number
            of returned labels. A choice of 5, will return the 5
            most probable labels. By default this returns only
            the most likely label and probability. threshold filters
            the returned labels by a threshold on probability. A
            choice of 0.5 will return labels with at least 0.5
            probability. k and threshold will be applied together to
            determine the returned labels.

        Args:
            batch (List[str]): list of text to get labels for.
            k (int, optional): number of labels to return for each text.
                Defaults to 1.
            threshold (float, optional): Minimum threshold for label probs.
                Defaults to 0.0.
            on_unicode_error (str, optional): Not sure.
                Defaults to "strict".

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of label and scores.
        """
        labels, probs = self.model.f.multilinePredict(
            batch, k, threshold, on_unicode_error
        )
        return np.array(labels).squeeze(), np.array(probs).squeeze()

    def lid_scores(
        self,
        series: pd.Series,
        batch_size: int = 1000,
        k: int = 2,
        threshold: float = 0.0,
        on_unicode_error: str = "strict",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Multiprocessing version of process_batch.

        args:
            series (pd.series):  list of text to get labels for.
            batch_size (int, optional): batch of text to send to fasttext.
                defaults to 1000.
            k (int, optional): number of labels to return for each text.
                defaults to 1.
            threshold (float, optional): Minimum threshold for label probs.
                Defaults to 0.0.
            on_unicode_error (str, optional): Not sure.
                Defaults to "strict".

        Returns:
            Tuple[np.ndarray, np.ndarray]: Labels, scores.

        """
        total_rows = len(series)
        print(total_rows)
        if total_rows < 1:
            return np.array([]), np.array([])

        def batch_generator(
            series_iterator: Iterable[tuple[Hashable, Any]], batch_size: int
        ):
            """
            Generate batches of data from the pd.Series.

            This is done because I don't want to convert
            the whole series into a list.
            """
            batch = []
            for _, text in series_iterator:  # still fast.
                batch.append(text)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

        with mp.Pool(
            initializer=self._init_worker, initargs=(self.model_path,)
        ) as pool:
            labels, scores = [], []
            with tqdm(total=total_rows, desc=self.pbar_label) as pbar:
                for batch_labels, batch_scores in pool.imap(
                    self._process_batch_wrapper,
                    (
                        (batch, k, threshold, on_unicode_error)
                        for batch in batch_generator(series.items(), batch_size)
                    ),
                ):
                    labels.append(batch_labels)
                    scores.append(batch_scores)
                    pbar.update(len(batch_labels))

        return np.concatenate(labels), np.concatenate(scores)

    @staticmethod
    def _init_worker(model_path):
        global worker_model  # to avoid pickling.
        worker_model = fasttext.load_model(model_path)

    @staticmethod
    def _process_batch_wrapper(args):
        """
        Function to call during multiprocessing.

        This is separate to process_batch since
        if we called process_batch() we would have to pickle
        the LID.
        """
        batch, k, threshold, on_unicode_error = args
        labels, probs = worker_model.f.multilinePredict(
            batch, k, threshold, on_unicode_error
        )
        return np.array(labels).squeeze(), np.array(probs).squeeze()
