import argparse
import codecs
import json
import sys

import numpy as np
import onmt
import torch

parser = argparse.ArgumentParser(description="preprocess.py")

##
## **Preprocess Options**
##

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

parser.add_argument(
    "-src_vocab_size", type=int, default=20000, help="Size of the source vocabulary"
)
parser.add_argument("-src_vocab", help="Path to an existing source vocabulary")

parser.add_argument("-seq_length", type=int, default=50, help="Maximum sequence length")
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

torch.manual_seed(opt.seed)


def makeVocabulary(filename, size):
    vocab = onmt.Dict(
        [
            onmt.Constants.PAD_WORD,
            onmt.Constants.UNK_WORD,
            onmt.Constants.BOS_WORD,
            onmt.Constants.EOS_WORD,
        ],
        lower=opt.lower,
        seq_len=opt.seq_length,
    )

    with codecs.open(filename, "r", "utf-8") as f:
        for sent in f.readlines():
            for word in sent.split():
                vocab.add(word)

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    print(
        "Created dictionary of size %d (pruned from %d)" % (vocab.size(), originalSize)
    )

    return vocab


def initVocabulary(name, dataFile, vocabFile, vocabSize):
    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print("Reading " + name + " vocabulary from '" + vocabFile + "'...")
        vocab = onmt.Dict()
        vocab.loadFile(vocabFile)
        print("Loaded " + str(vocab.size()) + " " + name + " words")

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print("Building " + name + " vocabulary...")
        genWordVocab = makeVocabulary(dataFile, vocabSize)

        vocab = genWordVocab

    print()
    return vocab


def saveVocabulary(name, vocab, file):
    print("Saving " + name + " vocabulary to '" + file + "'...")
    vocab.writeFile(file)


def makeData(srcFile, srcDicts):
    src, tgt = [], []
    sizes = []
    count, ignored = 0, 0

    print("Processing %s ..." % (srcFile))
    srcF = codecs.open(srcFile, "r", "utf-8")

    while True:
        sline = srcF.readline()

        # normal end of file
        if sline == "":
            break

        sline = sline.strip()
        ## source and/or target are empty
        if sline == "":
            print("WARNING: ignoring an empty line (" + str(count + 1) + ")")
            continue

        srcWords = sline.split()
        tgtWords = srcWords[0]
        srcWords = srcWords[1:]

        if len(srcWords) <= opt.seq_length and len(tgtWords) <= opt.seq_length:
            src += [
                srcDicts.convertToIdx(srcWords, onmt.Constants.UNK_WORD, padding=True)
            ]
            if tgtWords == opt.label0:
                tgt += [torch.LongTensor([0])]
            elif tgtWords == opt.label1:
                tgt += [torch.LongTensor([1])]
            sizes += [len(srcWords)]
        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print("... %d sentences prepared" % count)

    srcF.close()

    if opt.shuffle == 1:
        print("... shuffling sentences")
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    print("... sorting sentences by size")
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]

    print(
        "Prepared %d sentences (%d ignored due to length == 0 or > %d)"
        % (len(src), ignored, opt.seq_length)
    )

    return src, tgt


def main():
    dicts = {}
    print("Preparing source vocab ....")
    dicts["src"] = initVocabulary(
        "source", opt.train_src, opt.src_vocab, opt.src_vocab_size
    )

    print("Preparing training ...")
    train = {}
    train["src"], train["tgt"] = makeData(opt.train_src, dicts["src"])

    print("Preparing validation ...")
    valid = {}
    valid["src"], valid["tgt"] = makeData(opt.valid_src, dicts["src"])

    if opt.src_vocab is None:
        saveVocabulary("source", dicts["src"], opt.save_data + ".src.dict")

    print("Saving data to '" + opt.save_data + ".train.pt'...")
    save_data = {
        "dicts": dicts,
        "train": train,
        "valid": valid,
    }
    torch.save(save_data, opt.save_data + ".train.pt")


if __name__ == "__main__":
    main()
