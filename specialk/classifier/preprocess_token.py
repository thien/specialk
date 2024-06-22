import specialk.classifier.onmt as onmt
import argparse
import torch
import codecs
from typing import Union, List, Tuple, Iterable
from pathlib import Path
from tqdm import tqdm
from specialk.core.bpe import Encoder as BPEEncoder
from specialk.preprocess import parse as bpe_parse
import specialk.core.constants as BPEConstants
from specialk.core.utils import log


def get_args() -> argparse.Namespace:
    """Loads args.

    Returns:
        argparse.Namespace: Args from cli.
    """
    parser = argparse.ArgumentParser(description="preprocess.py")

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

    parser.add_argument(
        "-seq_length", type=int, default=50, help="Maximum sequence length"
    )
    parser.add_argument("-shuffle", type=int, default=1, help="Shuffle data")
    parser.add_argument("-seed", type=int, default=3435, help="Random seed")

    parser.add_argument("-bpe", action="store_true", help="If set, uses BPE encoding")
    parser.add_argument(
        "-pct_bpe", type=float, default=0.1, help="Percentage of tokens to use BPE"
    )
    parser.add_argument(
        "-load_vocab",
        action="store_true",
        help="If set, will only load vocabulary file. Error is raised if file does not exist.",
    )

    parser.add_argument("-lower", action="store_true", help="lowercase data")

    parser.add_argument(
        "-report_every",
        type=int,
        default=100000,
        help="Report status every this many sentences",
    )

    return parser.parse_args()


class Vocabulary:
    def __init__(self, name: str, filename: str, vocab_size: int):
        """
        Args:
            name (str): name of vocabulary.
            data_file (Union[Path,str]): path of file containing training data to use for the vocabulary.
            vocabulary_file (Union[Path,str]): path of vocabulary file to either load from, or save to.
            vocab_size (int, Optional): If set, sets cap on vocabulary size.

        Returns:
            onmt.Dict: Vocabulary file.
        """
        self.name = name
        self.filename = filename
        self.vocab_size = vocab_size
        self.vocab: onmt.Dict = onmt.Dict()

    def make(self, data_path: Union[Path, str]):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def to_tensor(self, text: str) -> torch.LongTensor:
        raise NotImplementedError

    def tokenize(text: str) -> List[str]:
        raise NotImplementedError


class BPEVocabulary(Vocabulary):
    def __init__(
        self, name: str, filename: str, vocab_size: int, seq_length: int, pct_bpe: float
    ):
        super().__init__(name, filename, vocab_size)
        self.seq_length = seq_length
        self.pct_bpe = pct_bpe
        self.vocab: BPEEncoder

    def make(self, data_path: Union[Path, str]) -> BPEEncoder:
        log.info("Creating BPEEncoder")
        src_bpe = BPEEncoder(
            vocab_size=self.vocab_size,
            pct_bpe=self.pct_bpe,
            ngram_min=1,
            UNK=BPEConstants.UNK_WORD,
            PAD=BPEConstants.PAD_WORD,
            word_tokenizer=bpe_parse,
        )
        src_bpe.fit(data_path)
        self.vocab = src_bpe
        log.info("Finished creating BPEEncoder")

    def to_tensor(self, text: Union[str, List[str]]) -> Iterable[List[int]]:
        """_summary_

        Args:
            text (Union[str, List[str]]): _description_

        Returns:
            Iterable[List[int]]: _description_
        """
        if isinstance(text, str):
            return self.vocab.transform([text])
        else:
            # it's a list
            return self.vocab.transform(text)

    def load(self):
        self.vocab.load(self.filename)

    def save(self):
        self.vocab.save(self.filename)

    def detokenize(self, tokens: List[int]) -> List[str]:
        raise NotImplementedError


class WordVocabulary(Vocabulary):
    """White-space level tokenization."""

    def __init__(
        self, name: str, filename: str, vocab_size: int, seq_length: int, lower: bool
    ):
        super().__init__(name, filename, vocab_size)
        self.vocab: onmt.Dict
        self.seq_length = seq_length
        self.lower = lower

    def make(self, data_path: Union[Path, str]):
        """
        Creates onmt dictionary of vocabulary.

        args:
            data_path: path of file to use (this is text data).
        """
        log.info("Creating WordVocabulary")
        vocab = onmt.Dict(
            [
                onmt.Constants.PAD_WORD,
                onmt.Constants.UNK_WORD,
                onmt.Constants.BOS_WORD,
                onmt.Constants.EOS_WORD,
            ],
            lower=self.lower,
            seq_len=self.seq_length,
        )

        with codecs.open(data_path, "r", "utf-8") as f:
            for sent in tqdm(f.readlines(), desc="Loading lines"):
                for word in self.tokenize(sent):
                    vocab.add(word)

        originalSize = vocab.size()

        # for debugging purposes, show the head distribution of the
        # token freuqencies.
        top_n = 10
        head_freq = sorted(
            vocab.labelToIdx.keys(),
            key=lambda label: vocab.frequencies[vocab.labelToIdx[label]],
            reverse=True,
        )[:top_n]
        log.debug(
            "Most frequent tokens found",
            tokens={k: vocab.frequencies[vocab.labelToIdx[k]] for k in head_freq},
            filepath=data_path,
        )

        self.vocab = vocab.prune(self.vocab_size)
        log.info(
            "Created space-separated token dictionary of size %d (pruned from %d)"
            % (self.vocab.size(), originalSize)
        )

    def load(self):
        if Path(self.filename).exists():
            # If given, load existing word dictionary.
            log.info(f"Reading {self.name} vocabulary from {self.filename}..")
            self.vocab.loadFile(self.filename)
            self.vocab_size = self.vocab.size()
            log.info(f"Loaded {self.vocab.size()} {self.name} tokens.")
        else:
            raise FileNotFoundError(f"{self.filename} doesn't exist.")

    def save(self):
        """Save vocabulary to file"""
        log.info("Saving " + self.name + " vocabulary to '" + self.filename + "'...")
        self.vocab.writeFile(self.filename)

    def to_tensor(self, text: str) -> torch.LongTensor:
        """Converts string to text to tensor of input.

        Args:
            text (str): Raw string of input.

        Returns:
            torch.LongTensor: List of indices corresponding to the token index from the vocab.
        """
        tokens = self.tokenize(text)
        return self.vocab.convertToIdx(tokens, onmt.Constants.UNK_WORD, padding=True)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Performs space level tokenization."""
        return [word for word in text.split()]

    def detokenize(self, tokens: List[torch.LongTensor]) -> List[str]:
        """Returns detokenized form (not concatenated though)"""
        return [self.vocab.idxToLabel[t.item()] for t in tokens]


def make_data(
    filepath: Union[Path, str],
    vocab: Vocabulary,
    label_0: str,
    label_1: str,
    shuffle: bool = True,
    sort: bool = True,
) -> Tuple[List[torch.LongTensor], List[torch.LongTensor]]:
    """Generates tokenized items.

    Args:
        filepath (Union[Path, str]): _description_
        vocab (Vocabulary): _description_
        shuffle (bool, optional): _description_. Defaults to True.
        sorted (bool, optional): _description_. Defaults to True.

    Returns:
        Tuple[List[torch.LongTensor, torch.LongTensor]]: src and target lists.
    """
    src, tgt = [], []
    sizes = []
    count, ignored = 0, 0

    log.info(f"Processing {filepath} ...")
    src_file = codecs.open(filepath, "r", "utf-8")

    PRINT_FIRST_LINE = False
    while True:
        sequence_line = src_file.readline()

        # normal end of file
        if sequence_line == "":
            break

        sequence_line = sequence_line.strip()
        ## source and/or target are empty
        if sequence_line == "":
            log.info("WARNING: ignoring an empty line (" + str(count + 1) + ")")
            continue

        src_tokens = sequence_line.split()
        src_tokens, tgt_tokens = " ".join(src_tokens[1:]), src_tokens[0]

        if not PRINT_FIRST_LINE:
            log.debug(
                "Printing first item in src_file",
                src_tokens=src_tokens,
                tgt_tokens=tgt_tokens,
                raw_line=sequence_line,
            )

        if len(src_tokens) <= vocab.seq_length and len(tgt_tokens) <= vocab.seq_length:
            src += [vocab.to_tensor(src_tokens)]
            if not PRINT_FIRST_LINE:
                sample_tokens = list(src[-1])
                log.debug(
                    "Printing tokenized line",
                    tokens=sample_tokens,
                    detokens=vocab.detokenize(sample_tokens),
                )
                PRINT_FIRST_LINE = True

            if tgt_tokens == label_0:
                tgt += [torch.LongTensor([0])]
            elif tgt_tokens == label_1:
                tgt += [torch.LongTensor([1])]
            sizes += [len(src_tokens)]
        else:
            ignored += 1

        count += 1

    src_file.close()

    if shuffle:
        log.info("... shuffling sentences")
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    if sort:
        log.info("... sorting sentences by size")
        _, perm = torch.sort(torch.Tensor(sizes))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]

    log.info(
        "Prepared %d sentences (%d ignored due to length == 0 or > %d)"
        % (len(src), ignored, vocab.seq_length)
    )

    return src, tgt


def main():
    opt = get_args()
    log.info("Loaded arguments", args=opt)

    torch.manual_seed(opt.seed)

    dicts = {}
    log.info("Preparing source vocab ....")

    if opt.bpe:
        log.info("Using BPE Tokenizer")
        tokenizer = BPEVocabulary(
            "source", opt.src_vocab, opt.src_vocab_size, opt.seq_length, opt.pct_bpe
        )
    else:
        log.info("Using space-separated Tokenizer")
        tokenizer = WordVocabulary(
            "source", opt.src_vocab, opt.src_vocab_size, opt.seq_length, opt.lower
        )
    vocabulary_file_exists = Path(opt.src_vocab).exists()
    if opt.load_vocab:
        if vocabulary_file_exists:
            log.info(
                "Path to vocabulary file is valid, attempting to load.",
                vocab_path=opt.src_vocab,
            )
            tokenizer.load()
        else:
            log.error(
                "Could not load vocabulary file, please check that the path to the file is valid.",
                vocab_path=opt.src_vocab,
            )
            raise FileNotFoundError
    else:
        log.info(
            "Could not load vocabulary file, generating new vocabulary.",
            vocab_path=opt.src_vocab,
        )
        tokenizer.make(opt.train_src)

    dicts["src"] = tokenizer.vocab

    log.info("Preparing training ...")
    train = {}
    train["src"], train["tgt"] = make_data(
        opt.train_src, tokenizer, opt.label0, opt.label1, opt.shuffle
    )

    log.info("Preparing validation ...")
    valid = {}
    valid["src"], valid["tgt"] = make_data(
        opt.valid_src, tokenizer, opt.label0, opt.label1, opt.shuffle
    )

    if not vocabulary_file_exists:
        tokenizer.save()

    log.info("Saving data to '" + opt.save_data + ".train.pt'...")
    save_data = {
        "dicts": dicts,
        "train": train,
        "valid": valid,
    }
    torch.save(save_data, opt.save_data + ".train.pt")


if __name__ == "__main__":
    main()
