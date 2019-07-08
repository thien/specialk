import spacy
import torch
import os
import json
from tqdm import tqdm
import argparse
import multiprocessing
import subprocess
from core.utils import get_len, batch_compute
from rouge import Rouge
import nltk
# import pandas as pd
# for emd
from pyemd import emd
from gensim.corpora.dictionary import Dictionary


class Metrics:
    """
    Handles performance measurements of either one tokenised
    document or comparisons between two tokenised documents.

    MAKE SURE THE DOCUMENTS ARE TOKENISED. (e.g. .atok files
    or model outputs.)
    """

    def __init__(self):
        self.running = True

    def load(self, args):
        self.load_glove(args.glove_path)
        self.load_spacy()
        self.load_syllables_dict()
        self.rouge = Rouge()
        return self

    def load_glove(self, glove_path):
        """
        Loads glove dataset.
        """
        # done some performance studies and found that this pandas method is faster than msgpack, json, and pickling.
        # also does not require creating a new file.
        df = pd.read_csv(glove_path, sep=" ", quoting=3, header=None, index_col=0)
        self.glove = {key: val.values for key, val in df.T.items()}
        return self

    def load_spacy(self):
        # load spacy model if it isn't in memory yet.
        if spacy not in dir(self):
            self.spacy = spacy.load('en_core_web_sm')
            self.sentencizer = self.spacy.create_pipe("sentencizer")
            self.spacy.add_pipe(self.sentencizer)
            self.tokenizer = self.spacy.create_pipe("tokenizer")
            self.stopwords = spacy.lang.en.stop_words.STOP_WORDS
        return self

    def load_syllables_dict(self):
        from nltk.corpus import cmudict
        self.cmudict = cmudict.dict()
        return self

    # NMT style measurements

    def wmd(self, ref_tokens, hyp_tokens):
        """
        Calculates word mover distances between two strings.

        Uses spacy tokenisers input.
        """
        # based on https://github.com/RaRe-Technologies/gensim/blob/18bcd113fd5ed31294a24c7274fcb95df001f88a/gensim/models/keyedvectors.py
        # If pyemd C extension is available, import it.
        # If pyemd is attempted to be used, but isn't installed, ImportError will be raised in wmdistance

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
            return float('inf')

        dictionary = Dictionary(documents=[document1, document2])
        vocab_len = len(dictionary)

        if vocab_len == 1:
            # Both documents are composed by a single unique token
            return 0.0

        # Sets for faster look-up.
        docset1, docset2 = set(document1),set(document2)

        # Compute distance matrix.
        distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
        for i, t1 in dictionary.items():
            if t1 not in docset1:
                continue
            for j, t2 in dictionary.items():
                if t2 not in docset2 or distance_matrix[i, j] != 0.0:
                    continue
                # Compute Euclidean distance between word vectors.
                distance_matrix[i, j] = distance_matrix[j, i] = np.sqrt(np.sum((glove[t1] - self.glove[t2])**2))

        if np.sum(distance_matrix) == 0.0:
            # `emd` gets stuck if the distance matrix contains only zeros.
            # logger.info('The distance matrix is all zeros. Aborting (returning inf).')
            return float('inf')

        def nbow(document):
            d = np.zeros(vocab_len, dtype=np.double)
            nbow = dictionary.doc2bow(document)  # Word frequencies.
            doc_len = len(document)
            for idx, freq in nbow:
                d[idx] = freq / float(doc_len)  # Normalized word frequencies.
            return d

        # Compute nBOW representation of documents.
        d1,d2 = nbow(document1),nbow(document2)

        # Compute WMD.
        return emd(d1, d2, distance_matrix)

    def bleu(self, ref_tokens, hyp_tokens):
        """
        chicken
        """
        # could try NLTK

        hypothesis = ['It', 'is', 'a', 'cat', 'at', 'room']
        reference = ['It', 'is', 'a', 'cat', 'inside', 'the', 'room']
        # there may be several references
        return nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)

    def rouge(self, ref_tokens, hyp_tokens):
        """
        Calculates ROUGE scores (uses an independent library.)
        """
        hypothesis = " ".join(hyp_tokens)
        reference = " ".join(ref_tokens)
        return rouge.get_scores(hypothesis, reference)

    def meteor(self, ref_tokens, hyp_tokens):
        
        # could try NLTK
        return None

    def perplexity(self, tokens):
        # not sure how to calculate this
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
            tokens: tokenized sentence from nlp(sentence)
        returns:
            matches: None if nothing is found,
                    [(match pairs)] otherwise.
        """

        if type(tokens) != spacy.tokens.doc.Doc:
            print("Warning: this is not a spacy processed input sequence. Manually processing..")
            tokens = self.spacy(tokens)

        matches = []

        index_limit = len(tokens)
        index = 0
        while index < index_limit:
            token = tokens[index]
            if token.text.lower() == "it":
                if tokens[index+1].pos_ == "VERB" and tokens[index+2].pos_ == "ADJ":
                    matches.append((index, (token, tokens[index+1], tokens[index+2])))
                    index = index + 2
            index += 1
            
        return matches if matches else None

    def lex_match_2(self, tokens):
        """
        finds ``v-link ADJ prep''
        
        eg:
            "..he was responsible for all for.."
        
        params:
            tokens: tokenized sentence from nlp(sentence)
        returns:
            matches: None if nothing is found,
                    [(match pairs)] otherwise.
        """
        if type(tokens) != spacy.tokens.doc.Doc:
            print("Warning: this is not a spacy processed input sequence. Manually processing..")
            tokens = self.spacy(tokens)

        matches = []

        index_limit = len(tokens)
        index = 0
        while index < index_limit:
            token = tokens[index]
            if token.pos_ == "VERB":
                group = [token]
                next_index = index+1
                # detect any adverbs before adj and adp.
                # e.g. "be *very, very,* embarrassing.."
                while tokens[next_index].pos_ == "ADV":
                    group.append(tokens[next_index])
                    next_index += 1
        
                if tokens[next_index].pos_ == "ADJ" and tokens[next_index+1].pos_ == "ADP":
                    group.append(tokens[next_index])
                    group.append(tokens[next_index+1])
                    matches.append((index, tuple(group)))
                    index = next_index + 2
            index += 1
            
        return matches if matches else None

    # readability measurements

    def syllables(self, word):
        """
        counts syllables in a word.

        returns int.
        """
        word = word.lower()
        if self.cmudict:
            if word in self.cmudict:
                return max([len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]])
        
        # imperfect but good enough calculation.
        # based on implementation from:
        # https://stackoverflow.com/questions/405161/detecting-syllables-in-a-word/4103234#4103234
        vowels = {"a","e","i","o","u","y"}
        numVowels = 0
        lastWasVowel = False
        for wc in word:
            foundVowel = False
            if wc in vowels:
                if not lastWasVowel:
                    numVowels += 1   # don't count diphthongs
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

    def readability(self, article):
        """
        Takes as input a string.
        Returns readability score of 0-100.
        """

        sentences = self.spacy(article).sents
        n_sents = 0
        n_words = 0
        n_sylls = 0
        n_chars = 0
        for sentence in sentences:
            n_sents += 1
            for word in self.tokenizer(str(sentence)):
                word = str(word).lower()
                num_syllables = self.syllables(word)
                if num_syllables > 0:
                    n_words += 1
                    n_sylls += num_syllables
                n_chars += len(word)

        reading_ease = 206.835 - 1.015 * (n_words/n_sents) - 84.6 * (n_sylls/n_words)
        grade_level = 0.39*(n_words/n_sents) + 11.8*(n_sylls/n_words)-15.59
        coleman_liau = 5.89 * (n_chars/n_words) - 0.3 * (n_sents/n_words) - 15.8
        # automated readability index
        ari = 4.71 * (n_chars/n_words) + 0.5*(n_words/n_sents) - 21.43

        return {
            "reading_ease" : reading_ease,
            "grade_level"  : grade_level,
            "coleman_liau" : coleman_liau,
            "ari" : ari
        }


def load_args():
    parser = argparse.ArgumentParser(description="metrics.py")
    # load documents
    parser.add_argument("-reference", required=True, type=str, 
                        help="""
                        filepath to reference text file containing
                        MOSES style sequences.
                        """)
    parser.add_argument("-ref_lang", required=True, type=str,
                        help="""
                        Reference document language.
                        """)
    parser.add_argument("-hypothesis", default=None, type=str, help="""
                        filepath to text file containing hypothesis
                        MOSES style sequences.
                        """)
    parser.add_argument("-hyp_lang", required=True, type=str,
                        help="""
                        Hypothesis document language.
                        """)

    # additional stuff

    parser.add_argument("-glove_path", type=str, help="""
                        Filepath to glove embedding weights.
                        """)


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


def load_dataset(reference_doc, hypothesis_doc=None):
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
                if len(line) < 1:
                    ignored.append(count)
                    continue
                sequences.append(line)

    if len(ignored) > 0:
        print("[Warning] There were",len(ignored),"ignored sequences.")
    return sequences


def batch_tokenise(sequence_pair):
    """
    Multithreaded tokenisation of sequences.
    """
    left, right = sequence_pair

    pass


if __name__ == "__main__":
    print("You shouldn't be running this directly.")
    args = load_args()
    dataset = load_dataset(args.reference, args.hypothesis)
    # load the metrics model
    metrics = Metrics()
    metrics.load_spacy()

    for seq_pair in dataset:
        if args.hypothesis_doc:
            reference, hypothesis = seq_pair
            reference = reference.split(" ")
            hypothesis = hypothesis.split(" ")
            if args.wmd:
                wmd = metrics.wmd(reference, hypothesis)
            if args.bleu:
                bleu = metrics.bleu(reference, hypothesis)
                print(bleu)
        break

    # output results to json.
    # calculate average.
    # metrics.load(args)