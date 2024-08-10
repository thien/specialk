"""
Various Wrapper functions for evaluation metrics, regarding text datasets.
"""

from pathlib import Path
from typing import List, Optional, Union

import gensim
import numpy as np
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm

from specialk.core.utils import hash
import specialk.metrics.style_transfer.cnn as cnn
import specialk.metrics.style_transfer.content_preservation as preserv
from specialk.metrics import AlignmentMetric, Metric
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
    """To use, init a style lexicon of the whole corpora, and then
        use that for test cases.
    """
    def __init__(
        self,
        dir_cache: Union[Path, str],
        dir_w2v: Optional[Union[Path, str]] = None,
        path_lexicon: Optional[Union[Path, str]] = None,
        enable_cache: bool = True,
    ):
        if isinstance(dir_cache, str):
            dir_cache = Path(dir_cache)
        if isinstance(dir_w2v, str):
            dir_w2v = Path(dir_w2v)
        if isinstance(path_lexicon, str):
            path_lexicon = Path(path_lexicon)

        self.cache_dir = dir_cache
        self.dir_w2v: Path = dir_w2v if dir_w2v else self.cache_dir / "embeddings"
        self.dir_lexicon: Path = (
            path_lexicon if path_lexicon else self.cache_dir / "style_lexicons"
        )

        self.style_lexicon = StyleLexicon()
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
        self, style_1: list[str], style_2: List[str], checksum: Optional[str] = None
    ):
        """Learn Style lexicon given the predictions and references.

        Args:
            prediction (str): Prediction text.
            reference (str): Reference text to compare to.
            checksum (Optional[str]): checksum representing the training data.
                If a checksum is passed in, then we'll load that dataset.
                Otherwise, we'll make a dataset, and save it.
        """

        # init style lexicon
        checksum = hash(style_1, style_2)

        path_style_lexicon: Path = self.dir_lexicon / checksum
        if path_style_lexicon.exists():
            self.style_lexicon = StyleLexicon.from_json(path_style_lexicon)
        else:
            self.style_lexicon.create(style_1, style_2)
            if self.enable_cache:
                self.style_lexicon.save(path_style_lexicon)
        return checksum

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
            checksum = hash(predictions, references)

        self.init_style_lexicon(predictions, references, checksum=checksum)

        # mask input and output texts
        src_masked = preserv.mark_style_words(
            predictions, style_tokens=self.style_lexicon.lexicon, mask_style=True
        )
        tgt_masked = preserv.mark_style_words(
            references, style_tokens=self.style_lexicon.lexicon, mask_style=True
        )

        # init w2v
        self.init_w2v(checksum, src_masked, tgt_masked)

        # calculate wmd scores
        return [
            self.word_movers_distance(src, tgt, self.w2v)
            for src, tgt in tqdm(zip(src_masked, tgt_masked))
        ]
