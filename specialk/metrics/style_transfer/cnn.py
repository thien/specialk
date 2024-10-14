import argparse
import codecs
import os
import sys

import torch
from tqdm import tqdm

import specialk.models.classifier.onmt as onmt

CNN_MODELS_DIR = "metrics/cnn_models"

NATURALNESS_DEFAULTS = {
    "political": os.path.join(CNN_MODELS_DIR, "naturalness_political.pt"),
    "publication": os.path.join(CNN_MODELS_DIR, "naturalness_publication.pt"),
}

FAKE_LABEL = 0
REAL_LABEL = 1

# assumes that you have already trained a model.


def addone(f):
    for line in f:
        yield line
    yield None


def setup_args(category, src):
    assert category in NATURALNESS_DEFAULTS
    opt = argparse.Namespace()
    opt.model = NATURALNESS_DEFAULTS[category]
    opt.label0 = FAKE_LABEL
    opt.label1 = REAL_LABEL
    opt.tgt = (
        REAL_LABEL  # we want the model to think the dataset is real. (but its fake!)
    )
    opt.src = src
    opt.batch_size = 128
    opt.max_sent_length = 50

    # is there a gpu?
    opt.cuda = False
    if opt.cuda:
        torch.cuda.set_device(0)
    return opt


def classify(category, src):
    opt = setup_args(category, src)
    model = onmt.Translator_cnn(opt)
    count = 0

    total_correct, total_words, total_loss = 0, 0, 0
    src_batch, tgt_batch = [], []
    outputs, predictions, sents = [], [], []

    for line in tqdm(src, desc="Naturalness"):
        count += 1
        if line is None and len(src_batch) == 0:
            break

        src_batch += [line.split()[: opt.max_sent_length]]
        tgt_batch += [opt.tgt]

        if len(src_batch) < opt.batch_size:
            continue

        n_correct, batch_size, outs, preds = model.translate(src_batch, tgt_batch)

        total_correct += n_correct.item()
        total_words += batch_size
        outputs += outs.data.tolist()
        predictions += preds.tolist()

        src_batch, tgt_batch = [], []

    return outputs
