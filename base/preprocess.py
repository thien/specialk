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
    assert formatting in ['word', 'character', 'bpe']

    if formatting == "character":
        return [i for i in text]
    return text.split()

def load_file(filepath, max_sequence_limit, formatting, case_sensitive=True, max_train_seq=None):
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
                # TODO: what's going on here?
                sequences.append([Constants.SOS_WORD, Constants.UNK_WORD, Constants.EOS_WORD])
            count += 1      
            if max_train_seq and count > max_train_seq:
                break

    print('[Info] Loaded {} sequences from {}'.format(len(sequences),filepath))

    if num_trimmed_sentences > 0:
        print('[Warning] Found {} sequences that needed to be trimmed to the maximum sequence length {}.'.format(num_trimmed_sentences, max_sequence_limit))
    return sequences

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
        Constants.UNK_WORD: Constants.UNK,
        Constants.BLO_WORD: Constants.BLO
    }

    # setup token conversions.
    words = sorted(vocabulary.keys(), key=lambda x:vocabulary[x], reverse=True)
    
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
    parser.add_argument('-save_name', required=True)
    parser.add_argument("-vocab_size", type=int, default=40000)
    parser.add_argument('-format', required=True, default='word', help="Determines whether to tokenise by word level, character level, or bytepair level.")
    parser.add_argument('-max_train_seq', default=None, type=int, help="""Determines the maximum number of training sequences.""")
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=50)
    parser.add_argument('-min_word_count', type=int, default=5, help="Minimum number of occurences before a word can be considered in the vocabulary.")
    parser.add_argument('-case_sensitive', action='store_true', help="Determines whether to keep it case sensitive or not.")
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
        src, src_bpe = load_file(dataset[g]['src'], opt.max_word_seq_len, opt.format, opt.case_sensitive, opt.max_train_seq)
        tgt, tgt_bpe = load_file(dataset[g]['tgt'], opt.max_word_seq_len, opt.format, opt.case_sensitive, opt.max_train_seq)
        if len(src) != len(tgt):
            print('[Warning] The {} sequence counts are not equal.'.format(g))
        # remove empty instances
        src,tgt = list(zip(*[(s, t) for s, t in zip(src, tgt) if s and t]))

        dataset[g]['src'], dataset[g]['tgt'] = src, tgt

    # build vocabulary
    bpe_enabled = opt.format.lower() == "bpe"
    if bpe_enabled:
        # building bpe vocabulary
        if opt.share_vocab:
            print('[Info] Building shared vocabulary for source and target sequences.')
            # build and train encoder
            corpus = dataset['train']['src'] + dataset['train']['tgt']
            bpe = bpe_encoder(vocab_size=4096, pct_bpe=0.8, ngram_min=1, UNK=Constants.UNK_WORD, PAD=Constants.PAD_WORD)
            bpe.fit(corpus)
            src_bpe, tgt_bpe = bpe
        else:
            print("[Info] Building voculabulary for source.")
            # build and train src and tgt encoder
            src_bpe = bpe_encoder(vocab_size=4096, pct_bpe=0.8, ngram_min=1, UNK=Constants.UNK_WORD, PAD=Constants.PAD_WORD)
            src_bpe.fit(dataset['train']['src'])
            tgt_bpe = bpe_encoder(vocab_size=4096, pct_bpe=0.8, ngram_min=1, UNK=Constants.UNK_WORD, PAD=Constants.PAD_WORD)
            tgt_bpe.fit(dataset['train']['tgt'])
        # translate sequences
        for g in tqdm(dataset, desc="Converting tokens into IDs"):
            for key in dataset[g]:
                if key == "src":
                    mthd = src_bpe
                else:
                    mthd = tgt_bpe
                mthd.mute()
                dataset[g][key] = [[f for f in mthd.transform([x])] for x in tqdm(dataset[g][key])]

    else:
        if opt.share_vocab:
            print('[Info] Building shared vocabulary for source and target sequences.')
            word2idx = build_vocabulary_idx(dataset['train']['src'] + dataset['train']['tgt'], opt.min_word_count, opt.vocab_size)
            src_word2idx, tgt_word2idx = word2idx
            print('[Info] Vocabulary size: {}'.format(len(word2idx)))
        else:
            print("[Info] Building voculabulary for source.")
            src_word2idx = build_vocabulary_idx(dataset['train']['src'], opt.min_word_count, opt.vocab_size)
            tgt_word2idx = build_vocabulary_idx(dataset['train']['tgt'], opt.min_word_count, opt.vocab_size)
            print('[Info] Vocabulary sizes -> Source: {}, Target: {}'.format(len(src_word2idx), len(tgt_word2idx)))
        # convert words in sequences to indexes.
        for g in tqdm(dataset, desc="Converting tokens into IDs"):
            for key in dataset[g]:
                if key == "src":
                    dataset[g][key] = seq2idx(dataset[g][key], src_word2idx)
                else:
                    dataset[g][key] = seq2idx(dataset[g][key], tgt_word2idx)
        # needs to be a method to throw away sequences with lots of UNK tokens.
        print()
        
        # setup data to store.
        data = {
            'settings' : opt,
            'dict' : {
                'src' : src_word2idx if not bpe_enabled else src_bpe.vocabs_to_dict(False),
                'tgt' : tgt_word2idx if not bpe_enabled else tgt_bpe.vocabs_to_dict(False)
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