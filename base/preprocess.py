import argparse
import subprocess
import unicodedata
from copy import deepcopy as copy
from functools import reduce

import core.constants as Constants
import torch
from core.bpe import Encoder as bpe_encoder
from core.utils import get_len
from tqdm import tqdm

"""
Preprocesses mose style code to pytorch ready files.
"""


def parse(text, formatting="word"):
    """
    text -> string of sentence.
    formatting -> one of 'word', 'character'.
    """
    assert type(text) == str
    assert formatting in ["word", "bpe"]

    if formatting == "word":
        return text.split()
    # otherwise default the response
    return text


def load_args():
    desc = """
    preprocess.py

    deals with convering the mose style datasets into data that can be interpreted by the neural models (for machine translation or style transfer).
    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-train_src", required=True)
    parser.add_argument("-train_tgt", required=True)
    parser.add_argument("-valid_src", required=True)
    parser.add_argument("-valid_tgt", required=True)
    parser.add_argument("-save_name", required=True)
    parser.add_argument("-vocab_size", type=int, default=35000)
    parser.add_argument("-pct_bpe", default=0.1)
    parser.add_argument(
        "-format",
        required=True,
        default="word",
        help="Determines whether to tokenise by word level, character level, or bytepair level.",
    )
    parser.add_argument(
        "-max_train_seq",
        default=None,
        type=int,
        help="""Determines the maximum number of training sequences.""",
    )
    parser.add_argument(
        "-max_valid_seq",
        default=None,
        type=int,
        help="""Determines the maximum number of validation sequences.""",
    )
    parser.add_argument("-max_len", "--max_word_seq_len", type=int, default=50)
    parser.add_argument(
        "-min_word_count",
        type=int,
        default=5,
        help="Minimum number of occurences before a word can be considered in the vocabulary.",
    )
    parser.add_argument(
        "-case_sensitive",
        action="store_true",
        help="Determines whether to keep it case sensitive or not.",
    )
    parser.add_argument("-share_vocab", action="store_true", default=False)
    parser.add_argument("-verbose", default=True, help="Output logs or not.")
    return parser.parse_args()


def build_vocabulary_idx(sentences, min_word_count, vocab_size=None):
    """
    Trims the vocabulary based by the frequency of words.

    Args:
    sentences: list of sentences (precomputed)
    min_word_count: cutoff parameter before a word/token is included in
                    the corpus vocabulary.

    returns a dictionary of word:id
    """
    # compute vocabulary and token counts.
    vocabulary = {}
    for sentence in tqdm(sentences, desc="Vocabulary Search", dynamic_ncols=True):
        for word in sentence:
            if word not in vocabulary:
                vocabulary[word] = 0
            vocabulary[word] += 1
    print("[Info] Original Vocabulary Size =", len(vocabulary))

    # setup dictionary.
    word2idx = {
        Constants.SOS_WORD: Constants.SOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK,
        Constants.BLO_WORD: Constants.BLO,
    }

    # setup token conversions.
    words = sorted(vocabulary.keys(), key=lambda x: vocabulary[x], reverse=True)

    num_ents = len(word2idx)
    ignored_word_count = 0
    for word in tqdm(words):
        if word not in word2idx:
            if vocabulary[word] > min_word_count:
                word2idx[word] = len(word2idx)
                num_ents += 1
                if vocab_size:
                    if num_ents == vocab_size:
                        break
            else:
                ignored_word_count += 1

    print(
        "[Info] Trimmed vocabulary size = {},".format(len(word2idx)),
        "each with minimum occurrence = {}".format(min_word_count),
    )
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx


def seq2idx(sequences, w2i):
    """
    Maps words to idx sequences.
    """
    return [[w2i.get(w, Constants.UNK) for w in s] for s in tqdm(sequences)]


def reclip(ref_seq, bpe_seq, enc, max_len):
    """
    Checks if the BPE encoded sequence is shorter than
    the max len. Otherwise, it'll trim the original
    sequence s.t. it fits within the max length of
    the sequence.
    """
    if len(bpe_seq) <= max_len:
        return bpe_seq

    subtract = 1
    while len(bpe_seq) > max_len:
        reference = " ".join(ref_seq.split()[:-subtract])
        subtract += 1
        bpe_seq = [x for x in enc.transform(reference)][0]
    # replace newly trimmed sequence.
    return bpe_seq


def load_file(filepath, formatting, case_sensitive=True, max_train_seq=None):
    """
    Loads text from file.
    """
    sequences = []

    count = 0
    with open(filepath) as f:
        for line in tqdm(f, total=get_len(filepath)):
            if not case_sensitive:
                line = line.lower()
            line = line.strip()
            sequences.append(line)
            count += 1
            if max_train_seq and count > max_train_seq:
                break

    print("[Info] Loaded {} sequences from {}".format(len(sequences), filepath))
    return sequences


if __name__ == "__main__":
    opt = load_args()
    # check if bpe is used.
    bpe_enabled = opt.format.lower() == "bpe"
    # setup the max token sequence length to include <s> and </s>
    opt.max_token_seq_len = opt.max_word_seq_len + 2

    # restructure code for readability
    dataset = {
        "train": {"src": opt.train_src, "tgt": opt.train_tgt},
        "valid": {"src": opt.valid_src, "tgt": opt.valid_tgt},
    }

    raw = copy(dataset)

    # load files
    for g in dataset:
        source, target = dataset[g]["src"], dataset[g]["tgt"]
        # setup limits
        if g == "train":
            num_seq_lim = opt.max_train_seq
        else:
            num_seq_lim = opt.max_valid_seq
        src = load_file(source, opt.format, opt.case_sensitive, num_seq_lim)
        tgt = load_file(target, opt.format, opt.case_sensitive, num_seq_lim)
        if len(src) != len(tgt):
            print("[Warning] The {} sequence counts are not equal.".format(g))
        # remove empty instances
        src, tgt = list(zip(*[(s, t) for s, t in zip(src, tgt) if s and t]))
        raw[g]["src"], raw[g]["tgt"] = src, tgt

    # learn vocabulary
    if bpe_enabled:
        # building bpe vocabulary
        if opt.share_vocab:
            print("[Info] Building shared vocabulary for source and target sequences.")
            # build and train encoder
            corpus = raw["train"]["src"] + raw["train"]["tgt"]
            bpe = bpe_encoder(
                vocab_size=opt.vocab_size,
                pct_bpe=opt.pct_bpe,
                ngram_min=1,
                UNK=Constants.UNK_WORD,
                PAD=Constants.PAD_WORD,
                word_tokenizer=parse,
            )

            bpe.fit(corpus)
            src_bpe, tgt_bpe = bpe
        else:
            print("[Info] Building voculabulary for source.")
            # build and train src and tgt encoder
            src_bpe = bpe_encoder(
                vocab_size=opt.vocab_size,
                pct_bpe=opt.pct_bpe,
                ngram_min=1,
                UNK=Constants.UNK_WORD,
                PAD=Constants.PAD_WORD,
                word_tokenizer=parse,
            )
            tgt_bpe = bpe_encoder(
                vocab_size=opt.vocab_size,
                pct_bpe=opt.pct_bpe,
                ngram_min=1,
                UNK=Constants.UNK_WORD,
                PAD=Constants.PAD_WORD,
                word_tokenizer=parse,
            )
            src_bpe.fit(raw["train"]["src"])
            tgt_bpe.fit(raw["train"]["tgt"])
    else:
        # note that we need to tokenise the sequences here.
        for g in dataset:
            source, target = raw[g]["src"], raw[g]["tgt"]
            source = [seq.split()[: opt.max_word_seq_len] for seq in source]
            target = [seq.split()[: opt.max_word_seq_len] for seq in target]
            dataset[g]["src"], dataset[g]["tgt"] = source, target

        del raw

        if opt.share_vocab:
            print("[Info] Building shared vocabulary for source and target sequences.")
            word2idx = build_vocabulary_idx(
                dataset["train"]["src"] + dataset["train"]["tgt"],
                opt.min_word_count,
                opt.vocab_size,
            )
            src_word2idx, tgt_word2idx = word2idx
            print("[Info] Vocabulary size: {}".format(len(word2idx)))
        else:
            print("[Info] Building voculabulary for source.")
            src_word2idx = build_vocabulary_idx(
                dataset["train"]["src"], opt.min_word_count, opt.vocab_size
            )
            tgt_word2idx = build_vocabulary_idx(
                dataset["train"]["tgt"], opt.min_word_count, opt.vocab_size
            )
            print(
                "[Info] Vocabulary sizes -> Source: {}, Target: {}".format(
                    len(src_word2idx), len(tgt_word2idx)
                )
            )

    # convert sequences
    if bpe_enabled:
        for g in tqdm(dataset, desc="Converting tokens into IDs"):
            for key in dataset[g]:
                bpe_method = src_bpe if key == "src" else tgt_bpe
                bpe_method.mute()
                dataset[g][key] = [f for f in bpe_method.transform(tqdm(raw[g][key]))]
    else:
        for g in tqdm(dataset, desc="Converting tokens into IDs"):
            for key in dataset[g]:
                method = src_word2idx if key == "src" else tgt_word2idx
                dataset[g][key] = seq2idx(dataset[g][key], method)

    # trim sequence lengths for bpe
    if bpe_enabled:
        for g in tqdm(dataset, desc="Trimming Sequences"):
            for key in dataset[g]:
                sequences = dataset[g][key]
                # it's much easier to just refer back to the original sentence and
                # trim tokens from there.
                bpe_method = src_bpe if key == "src" else tgt_bpe
                for i in range(len(sequences)):
                    ref_seq = raw[g][key][i]
                    bpe_seq = sequences[i]
                    dataset[g][key][i] = reclip(
                        ref_seq, bpe_seq, bpe_method, opt.max_word_seq_len
                    )
    # add <s>, </s>
    # (At this stage, all of the sequences are tokenised, so you'll need to input
    #  the ID values of SOS and EOS instead.)
    SOS, EOS = Constants.SOS, Constants.EOS
    for g in tqdm(dataset, desc="Adding SOS, EOS tokens"):
        for key in dataset[g]:
            dataset[g][key] = [[SOS] + x + [EOS] for x in dataset[g][key]]

    # setup data to save.
    data = {
        "settings": opt,
        "dict": {
            "src": src_word2idx if not bpe_enabled else src_bpe.vocabs_to_dict(False),
            "tgt": tgt_word2idx if not bpe_enabled else tgt_bpe.vocabs_to_dict(False),
        },
        "train": {"src": dataset["train"]["src"], "tgt": dataset["train"]["tgt"]},
        "valid": {"src": dataset["valid"]["src"], "tgt": dataset["valid"]["tgt"]},
    }

    # dump information.
    filename = opt.save_name + ".pt"
    print("[Info] Dumping the processed data to pickle file", filename)
    torch.save(data, filename)
    print("[Info] Done.")
