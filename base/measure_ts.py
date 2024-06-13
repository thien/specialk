
import argparse
import torch
import os
import sys
import json
from tqdm import tqdm
import multiprocessing
import numpy as np
import hashlib

# import nlgeval

# from pyemd import emd
from gensim.models.word2vec import Word2Vec
# from gensim.corpora.dictionary import Dictionary

# style transfer metrics codebase
import metrics.style_transfer.content_preservation as preserv
import metrics.style_transfer.style_lexicon as stme_lexicon
import metrics.style_transfer.utils as stme_utils
import metrics.style_transfer.tokenizer as stme_tokeniser

import metrics.style_transfer.cnn as cnn

from nltk.corpus import cmudict,stopwords
from nltk.translate.bleu_score import sentence_bleu
from nltk import pos_tag
from nltk import RegexpParser
from nltk import word_tokenize

sys.path.append('/home/t/Data/Files/Github/msc_project_model/base/')
from core.bpe import Encoder
from core.utils import batch_compute, get_len
from core.sentenciser import *

cachedir = "/home/t/Data/Datasets/msc_proj_cache/"

def load_args():
    parser = argparse.ArgumentParser(description="metrics.py")
    # load documents
    parser.add_argument("-src", required=True, type=str, 
                        help="""
                        filepath to reference text file containing
                        MOSES style sequences.
                        """)
    parser.add_argument("-tgt", default=None, type=str, help="""
                        filepath to text file containing hypothesis
                        MOSES style sequences.
                        """)

    parser.add_argument("-type", choices=['political', 'publication'], help="""
                        If enabled, runs newspaper specific lexical metrics.
                        """)
    parser.add_argument("-style_lexicon", default=None, type=str, help="""
                        Path to style lexicon json.
                        """)
    parser.add_argument("-cache_dir", default=cachedir, type=str, help="""
                        Path to cache dir.
                        """)
    parser.add_argument("-no_cache", default=True, help="""
                        If enabled, ignores cache dir.
                        """)
                        
    opt = parser.parse_args()
    return opt

class Measurements:
    def __init__(self, opt):
        self.opt = opt
        self.stopwords = set(stopwords.words('english'))
        self.cmudict = cmudict.dict()
        self.enable_cache = False
        if opt.cache_dir:
            self.init_cache_dir(opt.cache_dir)
        # working hash represents the hash of the dataset
        # we'll be working on.
        self.checksum = None 


    def init_cache_dir(self, cachedir):
        """
        Initiates directory and subdirectories
        for caches (a lot of this stuff is expensive to compute).
        """
        self.cachedir = cachedir
        if not os.path.exists(self.cachedir):
            os.makedirs(self.cachedir)
        subdirs = [
            "style_lexicons",
            "word2vec",
            "preservation"
        ]
        for foldername in subdirs:
            path = os.path.join(self.cachedir, foldername)
            if not os.path.exists(path):
                os.makedirs(path)
        self.enable_cache = True


    def intensity(self, src, tgt):
        # calculate bleu scores
        # calculate emd scores
        return None


    def naturalness(self, tgt, category):
        # runs a adversarial CNN that discriminates between
        # fake and real texts.
        
        # check if theres a model
        # if there isn't, then make one
        # load it
        # run it
        # return scores.
        return np.mean(cnn.classify(category, tgt))


    def preservation(self, src, tgt):
        """
        Computes preservation (Masked WMD) scores
        between the source and target sequences.
        """
        # check if we've cached it already.
        checksum = self.checksum if self.checksum else self.hash(src,tgt)
        if self.enable_cache:
            p_dir = os.path.join(self.cachedir, "preservation")
            results_path = os.path.join(p_dir,checksum)
            if checksum in os.listdir(p_dir):
                return stme_utils.load_json(results_path)

        lexicon = self.load_style_lexicon(src, tgt, self.opt.style_lexicon)  
        # train w2v model
        w2v = self.load_w2v(src, tgt)
        # mask input and output texts
        src_masked = preserv.mark_style_words(src, style_tokens=lexicon, mask_style=True) 
        tgt_masked = preserv.mark_style_words(tgt, style_tokens=lexicon, mask_style=True) 
        # calculate wmd scores
        emd_masked = self.calc_wmd(src_masked, tgt_masked, w2v)

        results = {
            'emd_masked' : np.mean(emd_masked)
        }

        # cache the wmd scores because they're quite
        # expensive to compute.
        if self.enable_cache:
            stme_utils.save_json(results, results_path)

        return results


    @staticmethod
    def calc_wmd(ref, cnd, w2v):
        n = len(ref)
        desc = "Calculating WMD"
        f = w2v.wv.wmdistance
        return [f(a,b) for a,b in tqdm(zip(ref,cnd), desc=desc, total=n)]


    def train_w2v(self, src, tgt):
        """
        Trains a style transfer word2vec s.t we can
        build a masked wmd comparator.
        """
        lexicon = self.load_style_lexicon(src, tgt, self.opt.style_lexicon)  
        masked_txt = preserv.mark_style_words(src+tgt, style_tokens=lexicon, mask_style=True) 
        return Word2Vec([stme_tokeniser.tokenize(x) for x in masked_txt])
    

    def load_w2v(self, src, tgt):
        """
        Loads working word2vec model if it exists,
        otherwise 
        """
        checksum = self.checksum if self.checksum else self.hash(src,tgt)

        if self.enable_cache:
            p_dir = os.path.join(self.cachedir, "word2vec")
            w2v_path = os.path.join(p_dir,checksum)
            if checksum in os.listdir(p_dir):
                model = Word2Vec.load(w2v_path)
                # normalize vectors
                model.init_sims(replace=True)
                return model

        # train the model otherwise
        w2v = self.train_w2v(src, tgt)

        if self.enable_cache:
            w2v.save(w2v_path)
        
        return w2v


    def create_lexicon(self, src, tgt):
        """
        finds style lexicon.
        Returns a set of words following a particular style.
        """
        styles = {0: 'styles'}
        # create vectoriser and inverse vocab.
        x, y = stme_utils.compile_binary_dataset(src, tgt)
        vectoriser = stme_lexicon.fit_vectorizer(x)
        inv_vocab = stme_utils.invert_dict(vectoriser.vocabulary_)
        # train style weights model
        src_weights = vectoriser.transform(x)
        model = stme_lexicon.train('l1', 3, src_weights, y)
        # extract style features and weights
        nz_weights, f_nums = stme_lexicon.extract_nonzero_weights(model)
        sf_and_weights = stme_lexicon.collect_style_features_and_weights(nz_weights, styles, inv_vocab, f_nums)
        # cache the results.
        return sf_and_weights
    

    def load_style_lexicon(self, src, tgt, lexicon_src=None):
        """
        Loads style lexicon if it was precomputed already.
        """
        if lexicon_src:
            return stme_utils.load_json(lexicon_src)
        else:
            # check if we've cached it already.
            checksum = self.checksum if self.checksum else self.hash(src,tgt)
            if self.enable_cache:
                p_dir = os.path.join(self.cachedir, "style_lexicons")
                lexicon_path = os.path.join(p_dir,checksum)
                if checksum in os.listdir(p_dir):
                    return stme_utils.load_json(lexicon_path)

            # otherwise, make it.
            lexicon = self.create_lexicon(src, tgt)
        
            if self.enable_cache:
                stme_utils.save_json(lexicon, lexicon_path)
        return lexicon


    @staticmethod
    def hash(src, tgt):
        # hashing function
        return str(hashlib.md5(str([x for x in zip(src,tgt)]).encode()).hexdigest())

# -----------------

def load_files(src, tgt):
    """
    loads src and tgt files.
    """
    with open(src) as a, open(tgt) as b:
        out = [x for x in tqdm(zip(a,b), total=get_len(src), desc="Files")]
    return [x[0] for x in out],[x[1] for x in out]

def express(opt):
    # load files
    src, tgt = load_files(opt.src, opt.tgt)
    # init measurements
    metrics = Measurements(opt)
    metrics.checksum = metrics.hash(src, tgt)

    # calculate preservation of meaning
    preservation = metrics.preservation(src, tgt)
    print("Preservation:", preservation)
    # calculate naturalness
    # naturalness = metrics.naturalness(tgt, opt.type)
    # print("Naturalness",naturalness)
    
    # if newspaper, then we can proceed to do newspaper
    # specific measurements.
    if opt.type == "publication":
        pass

if __name__ == "__main__":
    args = load_args()
    express(args)
