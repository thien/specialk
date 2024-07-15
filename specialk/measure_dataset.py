import argparse
import json
import multiprocessing
import os
import subprocess

# TODO: need to move spacy to outside s.t. it can be used for multiprocessing.
# TODO: need to move glove to outside for similar reasons to as spacy.
# TODO: same for syllables dict.
import warnings

import numpy as np
import pandas as pd
import spacy
import torch
from gensim.corpora.dictionary import Dictionary

# from specialk.metrics.legacy_meteor.meteor import Meteor
from nltk import RegexpParser, pos_tag, word_tokenize
from nltk.corpus import cmudict, stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.translate.bleu_score import sentence_bleu

# for emd
from pyemd import emd
from rouge import Rouge
from tqdm import tqdm

from specialk.core.sentenciser import *
from specialk.core.utils import batch_compute, get_len

warnings.filterwarnings("ignore")

# load rouge
rouge_comp = Rouge()
meteor = Meteor()


DEFAULT_GLOVE_PATH = "/home/t/Data/Datasets/glove/glove.6B.200d.txt"


class Metrics:
    """
    Handles additional performance measurements ontop
    of whatever nlg-eval does.
    Handles performance measurements of either one tokenised
    document or comparisons between two tokenised documents.

    MAKE SURE THE DOCUMENTS ARE TOKENISED. (e.g. .atok files
    or model outputs.)
    """

    def __init__(self, args):
        self.stopwords = set(stopwords.words("english"))
        self.cmudict = cmudict.dict()
        self.polarity = SentimentIntensityAnalyzer().polarity_scores

    def load(self, args):
        self.load_glove(args.glove_path)
        return self

    def load_glove(self, glove_path=DEFAULT_GLOVE_PATH):
        """
        Loads glove dataset.
        """
        # done some performance studies and found that this pandas
        # method is faster than msgpack, json, and pickling.
        # also does not require creating a new file.
        df = pd.read_csv(glove_path, sep=" ", quoting=3, header=None, index_col=0)
        self.glove = {key: val.values for key, val in df.T.items()}
        return self

    # preprocess

    def prep(self, sents, tokenise=True):
        """
        Preprocess sequence s.t. the same variable
        can be passed through all the metrics.
        """
        if type(sents) != str:
            # pair of sequences
            if tokenise:
                seqs = [x.split(" ") for x in sents]
                return seqs[0], seqs[1]
            return sents
        else:
            # single sequence
            if tokenise:
                seq = sents.split(" ")
                return seq

    @staticmethod
    def tokenize(x):
        return word_tokenize(x)

    @staticmethod
    def basic_stats(paragraphs):
        """
        Assumes the paragraphs are tokenised.
        """
        tokens = []
        sentence_lengths = []
        num_tokens_in_paragraph = []
        for paragraph in paragraphs:
            paragraph_tokens = []
            for sentence in paragraph:
                sentence_lengths.append(len(sentence))
                for token in sentence:
                    paragraph_tokens.append(token)
            tokens = tokens + paragraph_tokens
            num_tokens_in_paragraph.append(len(paragraph_tokens))

        num_tokens = len(tokens)
        num_paragraphs = len(paragraphs)
        num_sentences = len(sentence_lengths)

        avg_token_length = (
            sum([len(t) for t in tokens]) / num_tokens if num_tokens > 0 else 0
        )
        avg_sentence_length = sum(sentence_lengths) / num_sentences
        avg_paragraph_length = sum(num_tokens_in_paragraph) / len(paragraphs)

        return {
            "num_tokens": num_tokens,
            "num_paragraphs": num_paragraphs,
            "num_sentences": num_sentences,
            "avg_token_length": avg_token_length,
            "avg_sentence_length": avg_sentence_length,
            "avg_paragraph_length": avg_paragraph_length,
            "sentence_lengths": sentence_lengths,
        }

    # NMT style measurements

    def wmd(self, sequences):
        """
        Calculates word mover distances between two strings.

        Uses spacy tokenisers input.
        """
        # based on https://github.com/RaRe-Technologies/gensim/blob/18bcd113fd5ed31294a24c7274fcb95df001f88a/gensim/models/keyedvectors.py
        # If pyemd C extension is available, import it.
        # If pyemd is attempted to be used, but isn't installed, ImportError will be raised in wmdistance

        ref_tokens, hyp_tokens = self.prep(sequences, tokenise=True)

        documents = [ref_tokens, hyp_tokens]
        for i, document in enumerate(documents):
            # document = self.tokenizer(document)
            # remove stopwords.
            document = [i.lower() for i in document if not i in self.stopwords]
            # remove out-of-vocabulary words.
            document = [token for token in document if token in self.glove]
            documents[i] = document
        document1, document2 = documents

        if not document1 or not document2:
            # at least one of the documents had no words that were in the vocabulary.
            return float("inf")

        dictionary = Dictionary(documents=[document1, document2])
        vocab_len = len(dictionary)

        if vocab_len == 1:
            # Both documents are composed by a single unique token
            return 0.0

        # Sets for faster look-up.
        docset1, docset2 = set(document1), set(document2)

        # Compute distance matrix.
        distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
        for i, t1 in dictionary.items():
            if t1 not in docset1:
                continue
            for j, t2 in dictionary.items():
                if t2 not in docset2 or distance_matrix[i, j] != 0.0:
                    continue
                # Compute Euclidean distance between word vectors.
                distance_matrix[i, j] = distance_matrix[j, i] = np.sqrt(
                    np.sum((self.glove[t1] - self.glove[t2]) ** 2)
                )

        if np.sum(distance_matrix) == 0.0:
            # `emd` gets stuck if the distance matrix contains only zeros.
            # logger.info('The distance matrix is all zeros. Aborting (returning inf).')
            return float("inf")

        def nbow(document):
            d = np.zeros(vocab_len, dtype=np.double)
            nbow = dictionary.doc2bow(document)  # Word frequencies.
            doc_len = len(document)
            for idx, freq in nbow:
                d[idx] = freq / float(doc_len)  # Normalized word frequencies.
            return d

        # Compute nBOW representation of documents.
        d1, d2 = nbow(document1), nbow(document2)

        # Compute WMD.
        return {"value": emd(d1, d2, distance_matrix)}

    def bleu(self, sequences):
        """
        calculates BLEU score.
        """
        reference, hypothesis = self.prep(sequences, tokenise=True)
        # print(reference, hypothesis)
        bleu1 = sentence_bleu([reference], hypothesis, weights=(1, 0, 0, 0))
        bleu2 = sentence_bleu([reference], hypothesis, weights=(0, 1, 0, 0))
        bleu3 = sentence_bleu([reference], hypothesis, weights=(0, 0, 1, 0))
        bleu4 = sentence_bleu([reference], hypothesis, weights=(0, 0, 0, 1))

        return {"bleu_1": bleu1, "bleu_2": bleu2, "bleu_3": bleu3, "bleu_4": bleu4}

    def rouge(self, sequences):
        """
        Calculates ROUGE scores (uses an independent library.)
        """
        reference, hypothesis = self.prep(sequences, tokenise=False)
        return rouge_comp.get_scores(hypothesis, reference)[-1]

    def meteor(self, sequences):
        reference, hypothesis = self.prep(sequences, tokenise=False)
        # could try NLTK
        score = meteor.compute_score(hypothesis, reference)
        print(score)
        return None

    # style transfer intensity
    # content preservation
    # naturalness

    # lexical measurements

    def lex_match_1(self, tokens):
        """
        finds ``it v-link ADJ finite/non-finite clause''

        eg:
            "It's unclear what Teresa May is planning."

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

    def lex_match_2(self, tokens):
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

    # readability measurements

    def syllables(self, word):
        """
        counts syllables in a word.

        returns int.
        """
        word = word.lower()

        if word in self.cmudict:
            return max(
                [len(list(y for y in x if y[-1].isdigit())) for x in self.cmudict[word]]
            )

        # imperfect but good enough calculation.
        # based on implementation from:
        # https://stackoverflow.com/questions/405161/detecting-syllables-in-a-word/4103234#4103234
        vowels = {"a", "e", "i", "o", "u", "y"}
        numVowels = 0
        lastWasVowel = False
        for wc in word:
            foundVowel = False
            if wc in vowels:
                if not lastWasVowel:
                    numVowels += 1  # don't count diphthongs
                foundVowel = lastWasVowel = True
            if not foundVowel:
                # If full cycle and no vowel found,
                # set lastWasVowel to false
                lastWasVowel = False
        if len(word) > 2 and word[-2:] == "es":
            # Remove es - it's "usually" silent (?)
            numVowels -= 1
        elif len(word) > 1 and word[-1:] == "e":
            # remove silent e
            numVowels -= 1
        return numVowels if numVowels > 0 else 1

    def readability(self, article, tokenised=False):
        """
        Takes as input a string.
        Returns readability score of 0-100.
        """

        sentences = find_sentences(article)
        n_sents = 0
        n_words = 0
        n_sylls = 0
        n_chars = 0
        for sentence in sentences:
            n_sents += 1
            for word in self.tokenize(sentence):
                num_syllables = self.syllables(word.lower())
                if num_syllables > 0:
                    n_words += 1
                    n_sylls += num_syllables
                n_chars += len(word)

        reading_ease = (
            206.835 - 1.015 * (n_words / n_sents) - 84.6 * (n_sylls / n_words)
        )
        grade_level = 0.39 * (n_words / n_sents) + 11.8 * (n_sylls / n_words) - 15.59
        coleman_liau = 5.89 * (n_chars / n_words) - 0.3 * (n_sents / n_words) - 15.8
        # automated readability index
        ari = 4.71 * (n_chars / n_words) + 0.5 * (n_words / n_sents) - 21.43

        return {
            "reading_ease": reading_ease,
            "grade_level": grade_level,
            "coleman_liau": coleman_liau,
            "ari": ari,
        }


def load_args():
    parser = argparse.ArgumentParser(description="metrics.py")
    # load documents
    parser.add_argument(
        "-reference",
        required=True,
        type=str,
        help="""
                        filepath to reference text file containing
                        MOSES style sequences.
                        """,
    )
    parser.add_argument(
        "-ref_lang",
        required=True,
        type=str,
        help="""
                        Reference document language.
                        """,
    )
    parser.add_argument(
        "-hypothesis",
        default=None,
        type=str,
        help="""
                        filepath to text file containing hypothesis
                        MOSES style sequences.
                        """,
    )
    parser.add_argument(
        "-hyp_lang",
        default=None,
        type=str,
        help="""
                        Hypothesis document language.
                        """,
    )

    parser.add_argument(
        "-output",
        required=True,
        type=str,
        help="""
                        Location of output json filepath.
                        """,
    )
    # additional stuff

    parser.add_argument(
        "-glove_path",
        type=str,
        default=DEFAULT_GLOVE_PATH,
        help="""
                        Filepath to glove embedding weights.
                        """,
    )
    parser.add_argument(
        "-lowercase",
        action="store_true",
        help="""
                        If enabled, performs lowercase operation.
                        """,
    )

    # load measurement flags
    # these are automatically called from
    # the metrics function.
    mets = []
    for funct in dir(Metrics):
        if funct[0:4] == "load":
            continue
        if funct[0] == "_":
            continue
        funct_name = "-" + funct
        mets.append(funct_name)
    sorted(mets)
    for funct_name in mets:
        parser.add_argument(funct_name, action="store_true")

    opt = parser.parse_args()
    return opt


def load_dataset(reference_doc, hypothesis_doc=None, lowercase=False):
    """
    Since it is often the case that processing of the
    sequences is more expensive than loading the dataset,
    we load the whole datasets in memory first.
    """
    sequences = []
    ignored = []
    count = 0

    len_a = get_len(reference_doc)
    load_a = open(reference_doc, "r")

    if hypothesis_doc:
        len_b = get_len(hypothesis_doc)
        if len_a != len_b:
            print("[Warning] The datasets are not equal in length.")

        load_b = open(hypothesis_doc, "r")
        with load_a as a, load_b as b:
            for line in tqdm(zip(a, b), total=len_a):
                count += 1
                # remove trailing spaces
                line = [s.strip() for s in line]
                if lowercase:
                    line = [s.lower() for s in line]
                if len(line[0]) < 1 and len(line[1]) < 1:
                    ignored.append(count)
                    continue
                sequences.append(line)
    else:
        with load_a as a:
            for line in tqdm(a, total=len_a):
                count += 1
                # remove trailing spaces
                line = line.strip()
                if lowercase:
                    line = line.lower()
                if len(line) < 1:
                    ignored.append(count)
                    continue
                sequences.append(line)

    if len(ignored) > 0:
        print("[Warning] There were", len(ignored), "ignored sequences.")
    return sequences


if __name__ == "__main__":
    args = load_args()
    dataset = load_dataset(args.reference, args.hypothesis, args.lowercase)
    # setup ordering (for multiprocessing purposes)
    dataset = [(i, dataset[i]) for i in range(len(dataset))]
    # load the metrics model
    metrics = Metrics(args)

    def op_wrapper(seq):
        # # maintain the order of the sequences since we're
        # # doing batch operations.
        # i, cont = seq
        # return (i, operate(cont))
        idx, pair = seq
        src, out = pair
        readability = metrics.readability(out)
        polarity = metrics.polarity(out)
        s = pos_tag(word_tokenize(out))
        # print("POS:",s)
        lex_match_1 = metrics.lex_match_1(s)
        lex_match_2 = metrics.lex_match_2(s)
        return (
            idx,
            {
                "polarity": polarity,
                "readability": readability,
                "lex_match_1": lex_match_1,
                "lex_match_2": lex_match_2,
            },
        )

    results = batch_compute(op_wrapper, dataset)
    # sort by order and remove ordering tag
    results = sorted(results, key=lambda x: x[0])
    results = [x[1] for x in results]

    def calcmeans(lol):
        loldim = len(lol)
        keys = {"lex_match_1": 0, "lex_match_2": 0}
        readability_keys = {}
        polarity_keys = {}

        if "readability" in lol[0]:
            readability_keys = {k: 0 for k in lol[0]["readability"]}
        if "polarity" in lol[0]:
            polarity_keys = {k: 0 for k in lol[0]["polarity"]}

        skew_pol_count = 0

        for i in lol:
            for k in keys:
                if k[0:3] == "lex":
                    keys[k] += len(i[k])
            if len(readability_keys) > 0:
                for k in readability_keys:
                    readability_keys[k] += i["readability"][k]
            if len(polarity_keys) > 0:
                for k in polarity_keys:
                    if i["polarity"][k] in {0.0, 1.0}:
                        continue
                    polarity_keys[k] += i["polarity"][k]
                    skew_pol_count += 1

        keys = {k: keys[k] / loldim for k in keys}
        readability_keys = {k: readability_keys[k] / loldim for k in readability_keys}
        polarity_keys = {k: polarity_keys[k] / loldim for k in polarity_keys}

        return {**keys, **readability_keys, **polarity_keys}

    mean_stats = calcmeans(results)
    for k in mean_stats:
        print(k, mean_stats[k])

    torch.save(results, args.output)
