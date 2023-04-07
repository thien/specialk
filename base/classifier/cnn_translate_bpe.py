from __future__ import division

import onmt
import torch
import argparse
import math
import codecs
import sys
import os
from tqdm import tqdm
from copy import deepcopy as copy

sys.path.append("../")

import core.constants as Constants
from core.utils import get_len
from core.bpe import Encoder as bpe_encoder
from preprocess import seq2idx, reclip, load_file
from preprocess import parse as bpe_parse
import unicodedata

SOS, EOS = Constants.SOS, Constants.EOS

parser = argparse.ArgumentParser(description="cnn_translate_bpe.py")

parser.add_argument("-model", required=True, help="Path to model .pt file")
parser.add_argument("-num_classes", default=2, type=int, help="""Number of classes""")
parser.add_argument("-src", required=True, help="Source sequence to check")
parser.add_argument(
    "-tgt", required=True, help="The target label for the classifier check"
)
parser.add_argument("-label0", required=True, help="Label for 0")
parser.add_argument("-label1", required=True, help="Label for 1")
parser.add_argument(
    "-output",
    default=None,
    help="""Path to output the predictions (each line will
                    be the decoded sequence""",
)
parser.add_argument("-batch_size", type=int, default=30, help="Batch size")
parser.add_argument(
    "-verbose",
    action="store_true",
    help="Print scores and predictions for each sentence",
)
parser.add_argument("-gpu", type=int, default=-1, help="Device to run on")


def reportScore(name, scoreTotal, wordsTotal):
    print(
        "%s AVG SCORE: %.4f, %s PPL: %.4f"
        % (name, scoreTotal / wordsTotal, name, math.exp(-scoreTotal / wordsTotal))
    )


def addone(f):
    for line in f:
        yield line
    yield None


def main():
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    translator = onmt.Translator_cnn(opt)

    srcBatch, tgtBatch = [], []

    tgt_id = 0 if opt.tgt == opt.label0 else 1

    count = 0
    total_correct, total_words, total_loss = 0, 0, 0
    outputs, predictions, sents = [], [], []

    # load bpe encoder.
    bpe_enc = bpe_encoder.from_dict(translator.src_dict)
    bpe_enc.mute()

    max_seq_length = translator.model_opt.sequence_length

    for line in tqdm(
        addone(codecs.open(opt.src, "r", "utf-8")), total=get_len(opt.src) + 1
    ):
        count += 1
        if line is not None:
            sents.append(line)
            # tokenise.
            tokens = [f for f in bpe_enc.transform([line])][0]
            #             before = len(tokens)
            tokens = reclip(line, tokens, bpe_enc, max_seq_length - 2)
            #             after = len(tokens)
            tokens = [SOS] + tokens + [EOS]
            #             print("b, a:", before, after)
            # add padding.
            blanks = [Constants.PAD for _ in range(max_seq_length - len(tokens))]
            tokens = tokens + blanks

            srcBatch.append(tokens)

            tgtBatch += [tgt_id]

            if len(srcBatch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(srcBatch) == 0:
                break

        num_correct, batchSize, outs, preds = translator.translate_bpe(
            srcBatch, tgtBatch
        )

        total_correct += num_correct.item()
        total_words += batchSize
        outputs += outs.data.tolist()
        predictions += preds.tolist()

        srcBatch, tgtBatch = [], []
    if opt.output:
        with open(opt.output, "w") as outF:
            for i in range(len(sents)):
                outF.write(
                    str(predictions[i]) + "\t" + str(outputs[i]) + "\t" + sents[i]
                )

    print("Accuracy: ", str((total_correct * 100) / total_words))
    print("")


if __name__ == "__main__":
    main()
