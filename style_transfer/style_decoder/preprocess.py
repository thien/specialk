import onmt
import argparse
import torch
import codecs

parser = argparse.ArgumentParser(description='preprocess.py')

"""
`preprocess.py` deals with parsing the dataset such that we can feed them
into our models.
"""

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-train_src', required=True,
                    help="Path to the training source data")
parser.add_argument('-train_tgt', required=True,
                    help="Path to the training target data")
parser.add_argument('-valid_src', required=True,
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', required=True,
                     help="Path to the validation target data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=100000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=100000,
                    help="Size of the target vocabulary")
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")


parser.add_argument('-seq_length', type=int, default=50,
                    help="Maximum sequence length")
parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()

torch.manual_seed(opt.seed)

def makeVocabulary(filename, size):
    vocab = onmt.Dict([onmt.Constants.PAD_WORD,                                         onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD],
                       lower=opt.lower)

    with codecs.open(filename, "r", "utf-8") as f:
        for sent in f.readlines():
            for word in sent.split():
                vocab.add(word)

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFile, vocabFile, vocabSize):

    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = onmt.Dict()
        vocab.loadFile(vocabFile)
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFile, vocabSize)

        vocab = genWordVocab

    print()
    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(srcFile, tgtFile, srcDicts, tgtDicts):
    src, tgt, sizes = [], [], []
    count, ignored = 0, 0

    print('Processing %s & %s ...' % (srcFile, tgtFile))
    source_file = codecs.open(srcFile, "r", "utf-8")
    target_file = codecs.open(tgtFile, "r", "utf-8")

    while True:
        # read lines.
        source_line, target_line = source_file.readline(), target_file.readline()
        # if there's nothing on the lines then we're reached the end of file.
        if source_line == "" and target_line == "":
            break
        # source or target does not have same number of lines
        if source_line == "" or target_line == "":
            print('WARNING: source and target do not have the same number of sentences')
            break
        # remove trailing spaces.
        source_line, target_line = source_line.strip(),  target_line.strip()
        # source and/or target are empty
        if source_line == "" or target_line == "":
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            continue
        # split tokens by spaces. 
        source_tokens, target_tokens = source_line.split(), target_line.split()
        # check if the sentence is within range.
        if len(source_tokens) <= opt.seq_length and len(target_tokens) <= opt.seq_length:
            # add token sequences to dictionary.
            src += [srcDicts.convertToIdx(source_tokens,onmt.Constants.UNK_WORD)]
            tgt += [tgtDicts.convertToIdx(target_tokens,
                                          onmt.Constants.UNK_WORD,
                                          onmt.Constants.BOS_WORD,
                                          onmt.Constants.EOS_WORD)]
            sizes += [len(source_tokens)]
        else:
            ignored += 1
        # increment and show status when needed.
        count += 1
        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    source_file.close()
    target_file.close()

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src   = [src[idx]   for idx in perm]
        tgt   = [tgt[idx]   for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    print('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]

    print('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
          (len(src), ignored, opt.seq_length))

    return src, tgt

def main():
    dicts = {}
    print('Preparing source vocab ....')
    dicts['src'] = initVocabulary('source', opt.train_src, opt.src_vocab,
                                  opt.src_vocab_size)
    print('Preparing target vocab ....')
    dicts['tgt'] = initVocabulary('target', opt.train_tgt, opt.tgt_vocab,
                                  opt.tgt_vocab_size)

    print('Preparing training ...')
    train = {}
    train['src'], train['tgt'] = makeData(opt.train_src, opt.train_tgt,
                                          dicts['src'], dicts['tgt'])

    print('Preparing validation ...')
    valid = {}
    valid['src'], valid['tgt'] = makeData(opt.valid_src, opt.valid_tgt,
                                    dicts['src'], dicts['tgt'])

    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
    if opt.tgt_vocab is None:
        saveVocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')


    print('Saving data to \'' + opt.save_data + '.train.pt\'...')
    save_data = {'dicts': dicts,
                 'train': train,
                 'valid': valid,
                }
    torch.save(save_data, opt.save_data + '.train.pt')

if __name__ == "__main__":
    main()
