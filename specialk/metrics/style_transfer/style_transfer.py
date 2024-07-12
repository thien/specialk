"""
Various Wrapper functions for evaluation metrics, regarding text datasets.
"""

from specialk.metrics import Metric, AlignmentMetric

import hashlib
from pathlib import Path
from typing import List, Optional, Union

import gensim
import numpy as np
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm

import specialk.metrics.style_transfer.cnn as cnn
import specialk.metrics.style_transfer.content_preservation as preserv
from specialk.models.style_lexicon import StyleLexicon


class StyleMetric(Metric):
    """Umbrella object representing style-transfer associated metrics."""

    pass


class Intensity(StyleMetric, AlignmentMetric):
    def __init__(self, category: str):
        self.category = category
        assert self.category in {"political", "publication"}

    def compute(self, text: str) -> float:
        """Compute intensity score of given text.

        This runs a adversarial CNN that discriminates
        between political or publication classes.

        Args:
            text (str): document to perform classification against.

        Returns:
            float: value predicted by the model.
        """
        return np.mean(cnn.classify(self.category, text))


class Naturalness(StyleMetric):
    def compute(self, text: str) -> float:
        """Compute naturalness score of given text.

        This runs a adversarial CNN that discriminates between fake
        (computer generated) and real (written by humans) texts.

        Args:
            text (str): document to perform classification against.

        Returns:
            float: value predicted by the model.
        """
        return np.mean(cnn.classify("naturalness", text))


class Preservation(StyleMetric, AlignmentMetric):
    def __init__(
        self,
        dir_cache: str,
        dir_w2v: Optional[str] = None,
        path_lexicon: Optional[str] = None,
        enable_cache: bool = True,
    ):
        self.cache_dir = dir_cache
        self.dir_w2v = dir_w2v
        self.dir_lexicon = path_lexicon
        self.style_lexicon: StyleLexicon
        self.w2v: Word2Vec
        self.enable_cache = enable_cache

    @staticmethod
    def word_movers_distance(
        prediction: str,
        reference: str,
        word2vec: Union[
            gensim.models.Word2Vec, gensim.models.keyedvectors.KeyedVectors
        ],
    ) -> float:
        """Calculates Word Mover's Distance.

        Args:
            prediction (str): Prediction text.
            reference (str): Reference text to compare to.
            word2vec (Union[gensim.models.Word2Vec,
                            gensim.models.keyedvectors.KeyedVectors]):
                Vector representation of tokens to use for WMD.

        Returns:
           float: WMD score.
        """
        return word2vec.wv.wmdistance(prediction, reference)

    def init_style_lexicon(
        self, checksum: str, predictions: list[str], references: List[str]
    ):
        # init style lexicon
        path_style_lexicon: Path = self.dir_lexicon / checksum
        if path_style_lexicon.exists():
            self.style_lexicon = StyleLexicon.from_json(path_style_lexicon)
        else:
            self.style_lexicon.create(predictions, references)
            if self.enable_cache:
                self.style_lexicon.save(path_style_lexicon)

    def init_w2v(self, checksum: str, predictions: list[str], references: List[str]):
        path_w2v = self.dir_w2v / checksum
        if self.dir_w2v.exists():
            self.w2v: Word2Vec = Word2Vec.load(path_w2v)
        else:
            self.w2v = Word2Vec(predictions + references)
            if self.enable_cache:
                self.w2v.save(path_w2v)

    def compute(
        self,
        predictions: list[str],
        references: List[str],
        checksum: Optional[str] = None,
    ) -> List[float]:
        """Calculate preservation of meaning from style transfer.

        Compute preservation (Masked WMD) score between
        the source and target sequences.

        Args:
            predictions (list[str]): _description_
            references (Union[List[str], List[List[str]]]): _description_

        Raises:
            NotImplementedError: _description_
        """
        if not checksum:
            checksum = self.hash(predictions, references)

        self.init_style_lexicon(checksum, predictions, references)

        # mask input and output texts
        src_masked = preserv.mark_style_words(
            predictions, style_tokens=self.style_lexicon, mask_style=True
        )
        tgt_masked = preserv.mark_style_words(
            references, style_tokens=self.style_lexicon, mask_style=True
        )

        # init w2v
        self.init_w2v(checksum, src_masked, tgt_masked)

        # calculate wmd scores
        return [
            self.word_movers_distance(src, tgt, self.w2v)
            for src, tgt in tqdm(zip(src_masked, tgt_masked))
        ]

    @staticmethod
    def hash(src: list[str], tgt: list[str]) -> str:
        """Returns an md5 hash of the documents."""
        return str(hashlib.md5(str([x for x in zip(src, tgt)]).encode()).hexdigest())
