"""

preprocess_bpe is retrofitted to use BPE encodings for the
dataset.

"""

import argparse
import codecs
import json
import sys

import numpy as np
import torch

import specialk.classifier.onmt as onmt

sys.path.append("../")

import unicodedata
from copy import deepcopy as copy

from tqdm import tqdm

import specialk.core.constants as Constants
from specialk.core.bpe import Encoder as bpe_encoder
from specialk.preprocess import load_file, reclip, seq2idx
from specialk.preprocess import parse as bpe_parse


def load_args():
    parser = argparse.ArgumentParser(description="preprocess_bpe.py")

    parser.add_argument("-config", help="Read options from this file")

    parser.add_argument(
        "-train_src", required=True, help="Path to the training source data"
    )
    parser.add_argument("-label0", required=True, help="Label that would be 0")
    parser.add_argument("-label1", required=True, help="Label that would be 1")
    parser.add_argument(
        "-valid_src", required=True, help="Path to the validation source data"
    )

    parser.add_argument(
        "-save_data", required=True, help="Output file for the prepared data"
    )

    parser.add_argument("-vocab_size", type=int, default=35000)
    parser.add_argument("-pct_bpe", default=0.1)
    parser.add_argument(
        "-src_vocab", help="Path to an existing source vocabulary pt file."
    )

    parser.add_argument("-max_word_seq_len", type=int, default=50)
    parser.add_argument("-shuffle", type=int, default=1, help="Shuffle data")
    parser.add_argument("-seed", type=int, default=3435, help="Random seed")

    parser.add_argument("-lower", action="store_true", help="lowercase data")

    parser.add_argument(
        "-report_every",
        type=int,
        default=100000,
        help="Report status every this many sentences",
    )

    opt = parser.parse_args()

    return opt


def main():
    opt = load_args()
    bpe_enabled = True

    torch.manual_seed(opt.seed)

    # setup the max token sequence length to include <s> and </s>
    opt.max_token_seq_len = opt.max_word_seq_len

    # restructure code for readability
    dataset = {
        "train": {"src": opt.train_src, "tgt": []},
        "valid": {"src": opt.valid_src, "tgt": []},
    }

    label0, label1 = [torch.LongTensor([0])], [torch.LongTensor([1])]
    raw = copy(dataset)
    # load dataset
    for g in dataset:
        src = load_file(dataset[g]["src"], None, False)

        # split src and tgt
        src = [x.split() for x in src]
        tgt = [x[0] for x in src]
        src = [" ".join(x[1:]) for x in src]

        # convert tgt tokens.
        tgt = [label0 if i == opt.label0 else label1 for i in tgt]

        raw[g]["src"] = src
        dataset[g]["tgt"] = tgt

    if opt.src_vocab:
        # build bpe vocabulary
        print("[Info] Loading BPE vocabulary from", opt.src_vocab)
        src_bpe = bpe_encoder.from_dict(torch.load(opt.src_vocab)["dict"]["tgt"])
    else:
        # building bpe vocabulary
        print("[Info] Building BPE vocabulary.")
        # build and train encoder
        src_bpe = bpe_encoder(
            vocab_size=opt.vocab_size,
            pct_bpe=opt.pct_bpe,
            ngram_min=1,
            UNK=Constants.UNK_WORD,
            PAD=Constants.PAD_WORD,
            word_tokenizer=bpe_parse,
        )
        src_bpe.fit(raw["train"]["src"])

    # convert sequences
    for g in tqdm(dataset, desc="Converting tokens into IDs"):
        src_bpe.unmute()
        dataset[g]["src"] = [f for f in src_bpe.transform(tqdm(raw[g]["src"]))]

    for g in tqdm(dataset, desc="Trimming Sequences"):
        sequences = dataset[g]["src"]
        # it's much easier to just refer back to the original sentence and
        # trim tokens from there.
        for i in range(len(sequences)):
            ref_seq = raw[g]["src"][i]
            bpe_seq = sequences[i]
            dataset[g]["src"][i] = reclip(
                ref_seq, bpe_seq, src_bpe, opt.max_word_seq_len - 2
            )

    # add <s>, </s>
    # (At this stage, all of the sequences are tokenised, so you'll need to input
    #  the ID values of SOS and EOS instead.)
    SOS, EOS = Constants.SOS, Constants.EOS
    for g in tqdm(dataset, desc="Adding SOS, EOS tokens"):
        dataset[g]["src"] = [[SOS] + x + [EOS] for x in dataset[g]["src"]]

    # shuffle dataset by sizes
    for g in tqdm(dataset, desc="Shuffling and Sorting"):
        src, tgt = dataset[g]["src"], dataset[g]["tgt"]
        sizes = [len(x) for x in src]

        if opt.shuffle == 1:
            perm = torch.randperm(len(src))
            src = [src[idx] for idx in perm]
            tgt = [tgt[idx] for idx in perm]
            sizes = [sizes[idx] for idx in perm]

        _, perm = torch.sort(torch.Tensor(sizes))

        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]

        # add space to seq
        src_sizes = [sizes[idx] for idx in perm]

        blanks = [
            [Constants.PAD for _ in range(opt.max_token_seq_len - src_sizes[i])]
            for i in range(len(src))
        ]
        src = [src[i] + blanks[i] for i in range(len(src))]

        dataset[g]["src"] = src
        dataset[g]["tgt"] = tgt

    k = np.sum([len(x) for x in dataset[g]["src"]]) / len(dataset[g]["src"])
    print("len:", k)

    # setup data to save.
    data = {
        "settings": opt,
        "dicts": {"src": src_bpe.vocabs_to_dict(False)},
        "train": {"src": dataset["train"]["src"], "tgt": dataset["train"]["tgt"]},
        "valid": {"src": dataset["valid"]["src"], "tgt": dataset["valid"]["tgt"]},
    }

    # dump information.
    filename = opt.save_data + ".train.pt"
    print("[Info] Dumping the processed data to pickle file", filename)
    torch.save(data, filename)
    print("[Info] Done.")


if __name__ == "__main__":
    main()
