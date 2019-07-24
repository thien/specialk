"""
rebase.py

Used for setting up style transfer specific dataset
for use when training nmt models.
"""

import argparse
from tqdm import tqdm
import torch
import core.constants as Constants
from core.bpe import Encoder as bpe_encoder
from preprocess import load_file, seq2idx
from copy import deepcopy as copy

def load_args():
    parser = argparse.ArgumentParser(description="train.py")
    # data options
    parser.add_argument('-base', required=True,
                        help='path to the *.pt file that was computed through preprocess.py')
    parser.add_argument('-train_src', required=True)
    parser.add_argument('-train_tgt', required=True)
    parser.add_argument('-valid_src', required=True)
    parser.add_argument('-valid_tgt', required=True)
    parser.add_argument('-save_name', required=True)
    return parser.parse_args()

if __name__ == "__main__":
    opt = load_args()
    
    base = torch.load(opt.base)
    settings = base['settings']

    print(settings)

    is_bpe = settings.format.lower() == "bpe"

    dataset = {
        'train' : {
            'src' : opt.train_src,
            'tgt' : opt.train_tgt
        },
        'valid' : {
            'src' : opt.valid_src,
            'tgt' : opt.valid_tgt
        }
    }

    raw = copy(dataset)

    for g in dataset:
        source, target = raw[g]['src'], raw[g]['tgt']
        src = load_file(source, settings.format, settings.case_sensitive, settings.max_train_seq)
        tgt = load_file(target, settings.format, settings.case_sensitive, settings.max_train_seq)
        if len(src) != len(tgt):
            print('[Warning] The {} sequence counts are not equal.'.format(g))
        # remove empty instances
        src,tgt = list(zip(*[(s, t) for s, t in zip(src, tgt) if s and t]))
        raw[g]['src'], raw[g]['tgt'] = src, tgt

    # load sequence converters.
    if is_bpe:
        src_bpe = bpe_encoder.from_dict(base['dict']['src'])
        tgt_bpe = bpe_encoder.from_dict(base['dict']['tgt'])
    else:
        src_word2idx = base['dict']['src']
        tgt_word2idx = base['dict']['tgt']
        # tokenise
        for g in raw:
            source, target = raw[g]['src'], raw[g]['tgt']
            source = [seq.split()[:settings.max_word_seq_len] for seq in source]
            target = [seq.split()[:settings.max_word_seq_len] for seq in target]
            dataset[g]['src'], dataset[g]['tgt'] = source, target
        del raw

    # convert sequences
    if is_bpe:
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
    
    # add <s>, </s>
    for g in tqdm(dataset, desc="Adding SOS, EOS tokens"):
        SOS, EOS = Constants.SOS, Constants.EOS
        for key in dataset[g]:
            dataset[g][key] = [[SOS] + x + [EOS] for x in dataset[g][key]]

    # setup data to save.
    data = {
        'settings' : settings,
        'dict' : {
            'src' : src_word2idx if not is_bpe else src_bpe.vocabs_to_dict(False),
            'tgt' : tgt_word2idx if not is_bpe else tgt_bpe.vocabs_to_dict(False)
        },
        'train' : {
            'src' : dataset['train']['src'],
            'tgt' : dataset['train']['tgt']
        },
        'valid' : {
            'src' : dataset['valid']['src'],
            'tgt' : dataset['valid']['tgt']
        }
    }

    # dump information.
    filename = opt.save_name + ".pt"
    print('[Info] Dumping the processed data to pickle file', filename)
    torch.save(data, filename)
    print('[Info] Done.')
