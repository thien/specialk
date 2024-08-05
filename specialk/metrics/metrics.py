"""
Various Wrapper functions for evaluation metrics, regarding text datasets.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Union

import evaluate
import gensim.downloader as api
import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.models.keyedvectors import KeyedVectors
from nltk import pos_tag, word_tokenize
from nltk.corpus import cmudict, stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.translate.bleu_score import sentence_bleu
from pyemd import emd

from specialk.core.constants import PROJECT_DIR
from specialk.core.sentenciser import find_sentences
from specialk.core.utils import log


class Metric:
    """Metrics that measure properties of a sequence."""

    def compute(predictions: list[str]):
        raise NotImplementedError

    @lru_cache
    def tokenize(self, text: str) -> List[str]:
        return text.split(" ")


class AlignmentMetric(Metric):
    """Metrics that measure alignment between two sets of sequences."""

    def compute(predictions: list[str], references: Union[List[str], List[List[str]]]):
        raise NotImplementedError

    @staticmethod
    def use_first_references(
        references: Union[List[str], List[List[str]]],
    ) -> list[str]:
        if isinstance(references[0], list):
            # it's a list of lists.
            log.warn(
                "EMD can only consume one reference, but multiple are found."
                " Using only the first reference value."
            )
            new_references = []
            for reference in references:
                if isinstance(reference, list):
                    reference = reference[0]
                new_references.append(reference)
            return new_references
        return references


class LexicalMetrics(Metric):
    """Lexical Measurements against datasets."""

    def __init__(self):
        self.vowels = {"a", "e", "i", "o", "u", "y"}
        self.cmudict = cmudict.dict()

    def syllables(self, word: str) -> int:
        """Count syllables in a given word.

        Implementation is partially based on the following link:
        https://stackoverflow.com/questions/405161/detecting-syllables-in-a-word/4103234

        Args:
            word (str): word to count syllables from.

        Returns:
            int: number of syllables in a word.
        """
        word = word.lower()

        if word in self.cmudict:
            return max(
                [len(list(y for y in x if y[-1].isdigit())) for x in self.cmudict[word]]
            )

        # backup implementation, incase word is not found in cmu dictionary.
        num_vowels = 0
        prev_char_is_vowel = False
        for character in word:
            found_vowel = False
            if character in self.vowels:
                if not prev_char_is_vowel:
                    # don't count diphthongs
                    num_vowels += 1
                found_vowel = prev_char_is_vowel = True

            if not found_vowel:
                # If full cycle and no vowel found,
                # set prev_char_is_vowel to false.
                prev_char_is_vowel = False

        if len(word) > 2 and word[-2:] == "es":
            # Remove es - it's "usually" silent (?)
            num_vowels -= 1

        elif len(word) > 1 and word[-1:] == "e":
            # remove silent e
            num_vowels -= 1

        return num_vowels if num_vowels > 0 else 1

    @lru_cache
    def tokenize(self, text: str) -> List[str]:
        """Uses NLTK's recommended word tokenizer for word tokenization.
        This is because the downstream metrics will also leverage NLTK's libraries.

        Args:
            text (str): string of text.

        Returns:
            List[str]: Tokens of text.
        """
        return word_tokenize(text)

    def pos(self, text: str) -> List[Tuple[str, str]]:
        """Get POS Tags for given text.

        Use NLTK's currently recommended part of speech tagger to
        tag the given list of tokens.

        Args:
            text (str): _description_

        Returns:
            List[Tuple[str, str]]: _description_

        Example:
        >>> lexical_metric.pos("John's big idea isn't all that bad.")

        [('John', 'NOUN'), ("'s", 'PRT'), ('big', 'ADJ'), ('idea', 'NOUN'), ('is', 'VERB'),
        ("n't", 'ADV'), ('all', 'DET'), ('that', 'DET'), ('bad', 'ADJ'), ('.', '.')]
        """
        return pos_tag(self.tokenize(text))

    def lex_match_1(self, tokens: List[Tuple[str, str]]):
        """
        finds ``it v-link ADJ finite/non-finite clause''

        eg:
            "It's uncear what Teresa May is planning."

        params:
            tokens: pos tagged sequence (e.g. `pos_tag(word_tokenize(string_of_article))` )
        returns:
            matches: None if nothing is found,
                    [(match pairs)] otherwise.
        """

        index_limit = len(tokens)
        index = 0
        matches = []
        while index < index_limit:
            token, tag = tokens[index]

            if token.lower() == "it":
                if index + 2 < index_limit:
                    if tokens[index + 1][1][0] == "V":
                        if tokens[index + 2][1][0] == "J":
                            group = tuple([str(tokens[index + i][0]) for i in range(3)])
                            matches.append((index, group))
                        index = index + 2
                else:
                    break
            index += 1
        return matches

    def lex_match_2(self, tokens: List[Tuple[str, str]]):
        """
        finds ``v-link ADJ prep''

        eg:
            "..he was responsible for all for.."

        params:
            tokens: pos tagged sequence (e.g. `pos_tag(word_tokenize(string_of_article))` )
        returns:
            matches: None if nothing is found,
                    [(match pairs)] otherwise.
        """
        index_limit = len(tokens)
        index = 0
        matches = []
        while index < index_limit:
            token, tag = tokens[index]

            if tag[0] == "V":
                group = [token]
                next_index = index + 1
                # detect any adverbs before adj and adp.
                # e.g. "be *very, very,* embarrassing.."
                while (next_index < index_limit) and (tokens[next_index][1][0] == "R"):
                    group.append(tokens[next_index])
                    next_index += 1

                if next_index + 1 < index_limit:
                    if (
                        tokens[next_index][1][0] == "J"
                        and tokens[next_index + 1][1] == "IN"
                    ):
                        group.append(tokens[next_index][0])
                        group.append(tokens[next_index + 1][0])
                        matches.append((index, tuple(group)))
                        index = next_index + 2
            index += 1
        return matches

    def readability(self, article: str) -> Dict[str, Union[float, int]]:
        """Calculate readability scores from various metrics.

        Args:
            article (str): Block of text.
            tokenised (bool, optional): _description_. Defaults to False.

        Returns:
            Dict[str, Union[float, int]]: integer values are in the range of [0,100].
        """
        sentences: List[str] = find_sentences(article)
        n_sentences, n_words, n_syllables, n_characters = 0, 0, 0, 0

        for sentence in sentences:
            n_sentences += 1
            for word in self.tokenize(sentence):
                num_syllables = self.syllables(word.lower())
                if num_syllables > 0:
                    n_words += 1
                    n_syllables += num_syllables
                n_characters += len(word)

        READING_EASE = (
            206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (n_syllables / n_words)
        )
        GRADE_LEVEL = (
            0.39 * (n_words / n_sentences) + 11.8 * (n_syllables / n_words) - 15.59
        )
        COLEMAN_LIAU = (
            5.89 * (n_characters / n_words) - 0.3 * (n_sentences / n_words) - 15.8
        )
        # automated readability index
        ARI = 4.71 * (n_characters / n_words) + 0.5 * (n_words / n_sentences) - 21.43

        return {
            "reading_ease": READING_EASE,
            "grade_level": GRADE_LEVEL,
            "coleman_liau": COLEMAN_LIAU,
            "ari": ARI,
        }

    def basic_stats(self, article: List[str]) -> dict[str, float]:
        """Calculate basic statistics about an article.

        Args:
            paragraphs (List[str]): List of paragraphs, each are strings.

        Returns:
            dict[str, float]: various measurements against the dataset.
        """
        tokens: list[str] = []
        sentence_lengths: list[int] = []
        num_tokens_in_paragraph: list[int] = []
        sentence: str
        token: str

        for paragraph in article:
            paragraph_tokens = []

            for sentence in find_sentences(paragraph):
                sentence_lengths.append(len(sentence))
                for token in self.tokenize(sentence):
                    paragraph_tokens.append(token)
            tokens = tokens + paragraph_tokens
            num_tokens_in_paragraph.append(len(paragraph_tokens))

        num_tokens = len(tokens)
        num_paragraphs = len(article)
        num_sentences = len(sentence_lengths)

        avg_token_length = (
            sum([len(t) for t in tokens]) / num_tokens if num_tokens > 0 else 0
        )
        avg_sentence_length = sum(sentence_lengths) / num_sentences
        avg_paragraph_length = sum(num_tokens_in_paragraph) / len(article)

        return {
            "num_tokens": num_tokens,
            "num_paragraphs": num_paragraphs,
            "num_sentences": num_sentences,
            "avg_token_length": avg_token_length,
            "avg_sentence_length": avg_sentence_length,
            "avg_paragraph_length": avg_paragraph_length,
            "sentence_lengths": sentence_lengths,
        }


class Polarity(Metric):
    def __init__(self):
        self.polarity = SentimentIntensityAnalyzer().polarity_scores

    def compute(self, text: str) -> float:
        """Measure polarity of a given document.

        Return a float for sentiment strength based on the input text.
        Positive values are positive valence, negative value are negative
        valence.

        Hashtags are not taken into consideration (e.g. #BAD is neutral). If you
        are interested in processing the text in the hashtags too, then we recommend
        preprocessing your data to remove the #, after which the hashtag text may be
        matched as if it was a normal word in the sentence.

        Args:
            text (str): text to measure polarity against.

        Returns:
            float: Polarity score.
        """
        return self.polarity(text)


class Meteor(AlignmentMetric):
    def __init__(self):
        self.meteor = evaluate.load("meteor")

    def compute(
        self,
        predictions: List[str],
        references: Union[List[str], List[List[str]]],
        alpha=0.9,
        beta=3,
        gamma=0.5,
    ) -> float:
        """Calculate METEOR score.

        Args:
            predictions (List[str]): _description_
            references (Union[List[str], List[List[str]]]): _description_
            alpha (float, optional): _description_. Defaults to 0.9.
            beta (int, optional): _description_. Defaults to 3.
            gamma (float, optional): _description_. Defaults to 0.5.

        Returns:
            List[int]: _description_
        """
        return self.meteor.compute(
            predictions=predictions,
            references=references,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )["meteor"]


class BLEU(AlignmentMetric):
    @staticmethod
    def compute(
        predictions: List[str],
        references: Union[List[str], List[List[str]]],
    ) -> Dict[str, float]:
        bleu1 = sentence_bleu(references, [predictions], weights=(1, 0, 0, 0))
        bleu2 = sentence_bleu(references, [predictions], weights=(0, 1, 0, 0))
        bleu3 = sentence_bleu(references, [predictions], weights=(0, 0, 1, 0))
        bleu4 = sentence_bleu(references, [predictions], weights=(0, 0, 0, 1))

        return {"bleu_1": bleu1, "bleu_2": bleu2, "bleu_3": bleu3, "bleu_4": bleu4}


class SacreBLEU(AlignmentMetric):
    def __init__(self):
        self.bleu = evaluate.load("sacrebleu")

    def compute(
        self,
        predictions: List[str],
        references: Union[List[str], List[List[str]]],
    ) -> float:
        return self.bleu.compute(predictions=predictions, references=references)[
            "score"
        ]


class ChrF(AlignmentMetric):
    def __init__(self):
        self.chrf = evaluate.load("chrf")

    def compute(
        self,
        predictions: List[str],
        references: Union[List[str], List[List[str]]],
    ) -> float:
        return self.chrf.compute(predictions=predictions, references=references)[
            "score"
        ]


class ROUGE(AlignmentMetric):
    def __init__(self):
        self.rouge = evaluate.load("rouge")

    def compute(
        self,
        predictions: List[str],
        references: Union[List[str], List[List[str]]],
    ) -> dict[str, float]:
        """_summary_

        Args:
            predictions (List[str]): _description_
            references (Union[List[str], List[List[str]]]): _description_

        Returns:
            dict[str, float]: {'rouge1', 'rouge2', 'rougeL', 'rougeLsum'} values.
        """
        return self.rouge.get_scores(predictions=predictions, references=references)


class EarthMoverDistance(AlignmentMetric):
    def __init__(self):
        self.stopwords = set(stopwords.words("english"))
        self.glove: KeyedVectors = self.load_glove()

    def load_glove(self) -> KeyedVectors:
        """Load GloVe Word2Vec.

        Checks if the embeddings are already present in the cache_dir.

        Note that gensim.api.downloader downloads a text form of the
        embeddings, which takes a minute to load every time.

        We save a binary form of this model, which is an
        order of magnitude quicker to load than the text form.

        Returns:
            KeyedVectors: GloVe Embeddings.
        """
        cache_dir: Path = PROJECT_DIR / "cache" / "embeddings"
        path_w2v: Path = cache_dir / "w2v.bin"

        log.info("Loading GloVe Vectors")
        if path_w2v.exists():
            model = KeyedVectors.load_word2vec_format(path_w2v, binary=True)
            log.info("Loaded GloVe Vectors from cache.", path=str(path_w2v))
        else:
            # download glove from the internet, cache it as a binary.
            model: KeyedVectors = api.load("glove-twitter-200")
            cache_dir.mkdir(exist_ok=True)
            model.save(path_w2v, binary=True)
            log.info("Created GloVe cache.", path=str(path_w2v))
        return model

    @lru_cache
    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)

    def preprocess(self, tokens: List[str]) -> List[str]:
        """Run various preprocessing metrics.

        Removes stopwords, OOV.

        Args:
            tokens (List[str]): List of tokens.

        Returns:
            List[str]: Filtered list of tokens.
        """
        new_tokens = []
        for token in tokens:
            token = token.lower()
            if token in self.stopwords:
                continue
            if token not in self.glove:
                continue
            new_tokens.append(token)
        return new_tokens

    @staticmethod
    def nbow(document: List[str], dictionary: Dictionary) -> np.ndarray:
        """Create nBOW representation.

        Args:
            document (List[str]): tokenized document.
            dictionary (Dictionary): dictionary containing tokens from both src
            and target tokens.

        Returns:
            np.ndarray: vector representation of nBOW of the document.
        """

        vocab_len = len(dictionary)
        d = np.zeros(vocab_len, dtype=np.double)
        # Word frequencies.
        nbow = dictionary.doc2bow(document)
        doc_len = len(document)

        for idx, freq in nbow:
            # Normalize word frequencies.
            d[idx] = freq / float(doc_len)
        return d

    def compute(self, prediction: str, references: Union[str, List[str]]) -> float:
        """Calculates earth mover distances between two strings.

        based on https://github.com/RaRe-Technologies/gensim/blob/
          18bcd113fd5ed31294a24c7274fcb95df001f88a/gensim/models/keyedvectors.py
        If pyemd C extension is available, import it.
        If pyemd is attempted to be used, but isn't installed,
        ImportError will be raised in wmdistance.

        Args:
            predictions (List[str]): _description_
            references (Union[List[str], List[List[str]]]): _description_

        Returns:
            float: EarthMoversDistance, in the range of [0, inf].
        """
        references = self.use_first_references(references)

        prediction = self.preprocess(self.tokenize(prediction))
        references = self.preprocess(self.tokenize(references))

        if not prediction or not references:
            # at least one of the documents had no tokens that
            # were in the vocabulary.
            return float("inf")

        dictionary = Dictionary(documents=[prediction, references])
        vocab_len = len(dictionary)

        if vocab_len == 1:
            # Both documents are composed by a single unique token.
            return 0.0

        # Sets for faster look-up.
        set_pred, set_refs = set(prediction), set(references)

        # Compute distance matrix.
        distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
        for i, t1 in dictionary.items():
            if t1 not in set_pred:
                continue
            for j, t2 in dictionary.items():
                if t2 not in set_refs or distance_matrix[i, j] != 0.0:
                    continue
                # Compute Euclidean distance between word vectors.
                distance_matrix[i, j] = distance_matrix[j, i] = np.sqrt(
                    np.sum((self.glove[t1] - self.glove[t2]) ** 2)
                )

        if np.sum(distance_matrix) == 0.0:
            # `emd` gets stuck if the distance matrix contains only zeros.
            return float("inf")

        # Compute nBOW representation of documents.
        d1, d2 = self.nbow(prediction, dictionary), self.nbow(references, dictionary)

        # Compute WMD.
        return emd(d1, d2, distance_matrix)
