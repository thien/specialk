import argparse
import torch
import core.constants as Constants
from core.bpe import Encoder as bpe_encoder
import subprocess
from tqdm import tqdm
from functools import reduce

"""
Preprocesses mose style code to pytorch ready files.
"""

def get_num_seqs(filepath):
    """
    Counts number of lines in a given file.
    """
    command = "wc -l " + filepath
    process = subprocess.run(command.split(" "), stdout=subprocess.PIPE)
    return int(process.stdout.decode("utf-8").split(" ")[0])

def parse(text, formatting):
    """
    text -> string of sentence.
    formatting -> one of 'word', 'character'.
    """
    assert type(text) == str
    assert formatting in ['word', 'character']

    if formatting == "word":
        return text.split()
    if formatting == "character":
        return [i for i in text]

def load_file(filepath, max_sequence_limit, formatting, case_sensitive=True):
    """
    loads text from file.

    Args:
    filepath: location of moses formatted file to read.
    max_sequence_limit: maximum sequence length.
    formatting: see parse()
    case_sensitive: boolean.
    """
    sequences = []
    num_trimmed_sentences = 0

    breaker = 10
    count = 0 
    with open(filepath) as f:
        for sentence in tqdm(f, total=get_num_seqs(filepath)):
            if not case_sensitive:
                sentence = sentence.lower()
            words = parse(sentence, formatting)
            if len(words) > max_sequence_limit:
                num_trimmed_sentences += 1
            sequence = words[:max_sequence_limit]
            if sequence:
                sequence = [Constants.SOS_WORD] + sequence + [Constants.EOS_WORD]
                sequences.append(sequence)
            else:
                sequences.append(None)
            count += 1


    print('[Info] Loaded {} sequences from {}'.format(len(sequences),filepath))

    if num_trimmed_sentences > 0:
        print('[Warning] Found {} sequences that needed to be trimmed to the maximum sequence length {}.'.format(num_trimmed_sentences, max_sequence_limit))
    return sequences

def build_vocabulary_idx(sentences, min_word_count):
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
    for sentence in tqdm(sentences,desc="Vocabulary Search"):
        for word in sentence:
            if word not in vocabulary:
                vocabulary[word] = 0
            vocabulary[word] += 1
    print('[Info] Original Vocabulary Size =', len(vocabulary))

    # setup dictionary.
    word2idx = {
        Constants.SOS_WORD: Constants.SOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK
    }

    # setup token conversions.
    words = sorted(vocabulary.keys(), key=lambda x:vocabulary[x], reverse=True)
    
    ignored_word_count = 0
    for word in tqdm(words):
        if word not in word2idx:
            if vocabulary[word] > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1
    
    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx

def seq2idx(sequences, w2i):
    """
    Maps words to idx sequences.
    """
    return [[w2i.get(w, Constants.UNK) for w in s] for s in tqdm(sequences)]

def load_args():
    desc = """
    preprocess.py

    deals with convering the mose style datasets into data that can be interpreted by the neural models (for machine translation or style transfer).
    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-train_src', required=True)
    parser.add_argument('-train_tgt', required=True)
    parser.add_argument('-valid_src', required=True)
    parser.add_argument('-valid_tgt', required=True)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-format', required=True, default='word', help="Determines whether to tokenise by word level, character level, or bytepair level.")
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=50)
    parser.add_argument('-min_word_count', type=int, default=5, help="Minimum number of occurences before a word can be considered in the vocabulary.")
    parser.add_argument('-case_sensitive', action='store_true', default=True, help="Determines whether to keep it case sensitive or not.")
    parser.add_argument('-share_vocab', action='store_true', default=False)
    parser.add_argument('-verbose', default=True, help="Output logs or not.")
    return parser.parse_args()

if __name__ == "__main__":

    opt = load_args()

    # set up the max token sequence length to include <s> and </s>
    opt.max_token_seq_len = opt.max_word_seq_len + 2

    # restructure code for readability
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

    # load training and validation data.
    for g in dataset:
        src = load_file(dataset[g]['src'], opt.max_word_seq_len, opt.format, opt.case_sensitive)
        tgt = load_file(dataset[g]['tgt'], opt.max_word_seq_len, opt.format, opt.case_sensitive)
        if len(src) != len(tgt):
            print('[Warning] The {} sequence counts are not equal.'.format(g))
        # remove empty instances
        src,tgt = list(zip(*[(s, t) for s, t in zip(src, tgt) if s and t]))

        dataset[g]['src'], dataset[g]['tgt'] = src, tgt

    # build vocabulary
    if opt.share_vocab:
        print('[Info] Building shared vocabulary for source and target sequences.')
        word2idx = build_vocabulary_idx(dataset['train']['src'] + dataset['train']['tgt'], opt.min_word_count)
        src_word2idx, tgt_word2idx = word2idx
        print('[Info] Vocabulary size: {}'.format(len(word2idx)))
    else:
        print("[Info] Building voculabulary for source.")
        src_word2idx = build_vocabulary_idx(dataset['train']['src'], opt.min_word_count)
        tgt_word2idx = build_vocabulary_idx(dataset['train']['tgt'], opt.min_word_count)
        print('[Info] Vocabulary sizes -> Source: {}, Target: {}'.format(len(src_word2idx), len(tgt_word2idx)))
    # convert words in sequences to indexes.
    for g in tqdm(dataset, desc="Converting tokens into IDs"):
        for key in dataset[g]:
            if key == "src":
                dataset[g][key] = seq2idx(dataset[g][key], src_word2idx)
            else:
                dataset[g][key] = seq2idx(dataset[g][key], tgt_word2idx)
    print()
    
    # setup data to store.
    data = {
        'settings' : opt,
        'dict' : {
            'src' : src_word2idx,
            'tgt' : tgt_word2idx
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
    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data + ".pt")
    print('[Info] Done.')
