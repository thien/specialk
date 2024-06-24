import codecs
from pathlib import Path
from typing import Iterable, List, Union, Optional

import torch
from tqdm import tqdm

import specialk.classifier.onmt as onmt
from specialk.core.bpe import Encoder as BPEEncoder
from specialk.core.utils import log, load_dataset
from specialk.preprocess import parse as bpe_parse
import specialk.core.constants as Constants


class Vocabulary:
    def __init__(
        self,
        name: str,
        filename: str,
        vocab_size: int,
        max_length: int,
        BOS_TOKEN: Optional[str] = Constants.SOS_WORD,
        EOS_TOKEN: Optional[str] = Constants.EOS_WORD,
        UNK_TOKEN: Optional[str] = Constants.UNK_WORD,
        SEP_TOKEN: Optional[str] = Constants.UNK_WORD,
        PAD_TOKEN: Optional[str] = Constants.SEP_WORD,
        CLS_TOKEN: Optional[str] = Constants.CLS_TOKEN,
        BLO_TOKEN: Optional[str] = Constants.BLO_WORD,
    ):
        """
        Args:
            name (str): name of vocabulary.
            data_file (Union[Path,str]): path of file containing training data to use for the vocabulary.
            vocabulary_file (Union[Path,str]): path of vocabulary file to either load from, or save to.
            vocab_size (int, Optional): If set, sets cap on vocabulary size.
            fixed_length (int): maxiumum token length of a sequence.
            BOS_TOKEN (Optional[str], optional): A special token representing the beginning of a sentence. Defaults to Constants.SOS_WORD.
            EOS_TOKEN (Optional[str], optional): A special token representing the end of a sentence. Defaults to Constants.EOS_WORD.
            UNK_TOKEN (Optional[str], optional):  A special token representing an out-of-vocabulary token. Defaults to Constants.UNK_WORD.
            SEP_TOKEN (Optional[str], optional): A special token separating two different sentences in the same input (used by BERT for instance).. Defaults to Constants.UNK_WORD.
            PAD_TOKEN (Optional[str], optional): A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by attention mechanisms or loss computation. Defaults to Constants.SEP_WORD.
            CLS_TOKEN (Optional[str], optional):  A special token representing the class of the input (used by BERT for instance). Defaults to Constants.CLS_TOKEN.
            BLO_TOKEN (Optional[str], optional):  A special token representing the separation of paragraph blocks. Defaults to Constants.BLO_WORD.
        Returns:
            onmt.Dict: Vocabulary file.
        """
        self.name = name
        self.filename = filename
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.vocab: onmt.Dict = onmt.Dict()

        self.BOS_TOKEN = BOS_TOKEN
        self.EOS_TOKEN = EOS_TOKEN
        self.UNK_TOKEN = UNK_TOKEN
        self.SEP_TOKEN = SEP_TOKEN
        self.PAD_TOKEN = PAD_TOKEN
        self.CLS_TOKEN = CLS_TOKEN
        self.BLO_TOKEN = BLO_TOKEN

    def make(self, data_path: Union[Path, str]):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def save(self, filepath: Optional[Union[Path, str]] = None):
        if not filepath:
            filepath = self.filename
        raise NotImplementedError

    def to_tensor(self, text: str) -> torch.LongTensor:
        raise NotImplementedError

    def tokenize(text: str) -> List[str]:
        raise NotImplementedError


class BPEVocabulary(Vocabulary):
    def __init__(
        self, name: str, filename: str, vocab_size: int, max_length: int, pct_bpe: float
    ):
        super().__init__(name, filename, vocab_size, max_length=max_length)
        self.pct_bpe = pct_bpe
        self.vocab: BPEEncoder

    def make(self, data_path: Union[Path, str]) -> BPEEncoder:
        log.info("Creating BPEEncoder.")
        src_bpe = BPEEncoder(
            vocab_size=self.vocab_size,
            pct_bpe=self.pct_bpe,
            ngram_min=1,
            UNK=self.UNK_TOKEN,
            PAD=self.PAD_TOKEN,
            word_tokenizer=bpe_parse,
        )
        # loading data path
        dataset: List[str] = load_dataset(data_path)
        src_bpe.fit(dataset)
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
            return list(self.vocab.transform([text], fixed_length=self.max_length))
        else:
            # it's a list
            return list(self.vocab.transform(text, fixed_length=self.max_length))

    def load(self):
        self.vocab = BPEEncoder.load(self.filename)

    def save(self, filepath: Optional[Union[Path, str]] = None):
        if not filepath:
            filepath = self.filename
        log.info(f"Saving vocabulary '{self.name}' to '{filepath}'...")

        self.vocab.save(filepath)

    def detokenize(self, tokens: List[int]) -> List[str]:
        raise NotImplementedError

    def tokenize(self, text: str) -> List[str]:
        return self.vocab.tokenize(text)


class WordVocabulary(Vocabulary):
    """White-space level tokenization."""

    def __init__(
        self, name: str, filename: str, vocab_size: int, max_length: int, lower: bool
    ):
        super().__init__(name, filename, vocab_size, max_length=max_length)
        self.vocab: onmt.Dict
        self.lower = lower
        self.PAD_TOKEN = onmt.Constants.PAD_WORD
        self.UNK_TOKEN = onmt.Constants.UNK_WORD
        self.BOS_TOKEN = onmt.Constants.BOS_WORD
        self.EOS_TOKEN = onmt.Constants.EOS_WORD

    def make(self, data_path: Union[Path, str]):
        """
        Creates onmt dictionary of vocabulary.

        args:
            data_path: path of file to use (this is text data).
        """
        log.info("Creating WordVocabulary")
        vocab = onmt.Dict(
            [
                self.PAD_TOKEN,
                self.UNK_TOKEN,
                self.BOS_TOKEN,
                self.EOS_TOKEN,
            ],
            lower=self.lower,
            seq_len=self.max_length,
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
            self.seq_len = self.max_length
            self.lower = self.lower
            log.info(f"Loaded {self.vocab.size()} {self.name} tokens.")
        else:
            raise FileNotFoundError(f"{self.filename} doesn't exist.")

    def save(self, filepath: Optional[Union[Path, str]] = None):
        if not filepath:
            filepath = self.filename
        """Save vocabulary to file"""
        log.info(f"Saving vocabulary '{self.name}' to '{filepath}'...")

        self.vocab.writeFile(filepath)

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
