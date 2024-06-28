import argparse
import hashlib
import json
import multiprocessing
import os
import sys

import numpy as np
import torch
from typing import Union, Optional
from pathlib import Path

# import nlgeval
# from pyemd import emd
from gensim.models.word2vec import Word2Vec
from nltk import RegexpParser, pos_tag, word_tokenize
from nltk.corpus import cmudict, stopwords
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

import specialk.metrics.style_transfer.cnn as cnn
from specialk.core.utils import log

# from gensim.corpora.dictionary import Dictionary
# style transfer metrics codebase

# sys.path.append('/home/t/Data/Files/Github/msc_project_model/base/')
from specialk.core.sentenciser import *
from specialk.core.utils import batch_compute, get_len, log
from specialk.metrics import Preservation, Naturalness, Intensity

cachedir = "/home/t/Data/Datasets/msc_proj_cache/"


def load_args():
    parser = argparse.ArgumentParser(description="metrics.py")
    # load documents
    parser.add_argument(
        "-src",
        required=True,
        type=str,
        help="""
                        filepath to reference text file containing
                        MOSES style sequences.
                        """,
    )
    parser.add_argument(
        "-tgt",
        default=None,
        type=str,
        help="""
                        filepath to text file containing hypothesis
                        MOSES style sequences.
                        """,
    )

    parser.add_argument(
        "-type",
        choices=["political", "publication"],
        help="""
                        If enabled, runs newspaper specific lexical metrics.
                        """,
    )
    parser.add_argument(
        "-style_lexicon",
        default=None,
        type=str,
        help="""
                        Path to style lexicon json.
                        """,
    )
    parser.add_argument(
        "-cache_dir",
        default=cachedir,
        type=str,
        help="""
                        Path to cache dir.
                        """,
    )
    parser.add_argument(
        "-no_cache",
        default=True,
        help="""
                        If enabled, ignores cache dir.
                        """,
    )

    opt = parser.parse_args()
    return opt


class Measurements:
    def __init__(self, opt):
        self.opt = opt
        self.enable_cache = False
        if opt.cache_dir:
            self.init_cache_dir(opt.cache_dir)
        # working hash represents the hash of the dataset
        # we'll be working on.
        self.checksum = None

        self.presevation = Preservation(self.cachedir)
        self.naturalness = Naturalness(self.cachedir)
        self.intensity = Intensity(self.cachedir)

    def init_cache_dir(self, cachedir: str):
        """
        Initiates directory and subdirectories
        for caches (a lot of this stuff is expensive to compute).
        """
        self.cachedir = cachedir

        if not os.path.exists(self.cachedir):
            log.info("Making cache dir:", self.cachedir)
            os.makedirs(self.cachedir)

        subdirs = ["style_lexicons", "word2vec", "preservation"]
        for foldername in subdirs:
            path = os.path.join(self.cachedir, foldername)
            os.makedirs(path, exist_ok=True)

        self.enable_cache = True

    @staticmethod
    def hash(src: list[str], tgt: list[str]) -> str:
        """Returns an md5 hash of the documents."""
        return str(hashlib.md5(str([x for x in zip(src, tgt)]).encode()).hexdigest())


# -----------------


def load_files(src: str, tgt: str):
    """
    loads src and tgt files.
    """
    with open(src) as a, open(tgt) as b:
        out = [x for x in tqdm(zip(a, b), total=get_len(src), desc="Files")]
    return [x[0] for x in out], [x[1] for x in out]


def express(opt):
    # load files
    src, tgt = load_files(opt.src, opt.tgt)
    # init measurements
    metrics = Measurements(opt)
    metrics.checksum = metrics.hash(src, tgt)

    # calculate preservation of meaning
    preservation = metrics.preservation.compute(src, tgt, metrics.checksum)
    avg_preservation = np.mean(preservation)
    log.info(f"Preservation: {avg_preservation}")
    # calculate naturalness
    naturalness = metrics.naturalness.compute(tgt, opt.type)
    log.info(f"Naturalness: {naturalness}")

    # if newspaper, then we can proceed to do newspaper
    # specific measurements.
    if opt.type == "publication":
        pass


if __name__ == "__main__":
    args = load_args()
    log.info("Loaded args", args=args)
    express(args)
