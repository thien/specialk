"""Tokenizer/Vocabulary library."""

from __future__ import annotations
import codecs
from pathlib import Path
from typing import Iterable, List, Optional, Union, Any
import json

import torch
from sacremoses import MosesDetokenizer, MosesTokenizer
from tqdm import tqdm

import specialk.models.classifier.onmt as onmt
import specialk.core.constants as Constants
from specialk.core.bpe import Encoder as BPEEncoder
from specialk.core.utils import load_dataset, log, deprecated
from specialk.datasets.preprocess import parse as bpe_parse

import sentencepiece as spm


class Vocabulary:
    def __init__(
        self,
        name: str,
        vocab_size: int,
        max_length: int,
        filename: str = "",
        BOS_TOKEN: Optional[str] = Constants.SOS_WORD,
        EOS_TOKEN: Optional[str] = Constants.EOS_WORD,
        UNK_TOKEN: Optional[str] = Constants.UNK_WORD,
        SEP_TOKEN: Optional[str] = Constants.SEP_WORD,
        PAD_TOKEN: Optional[str] = Constants.PAD_WORD,
        CLS_TOKEN: Optional[str] = Constants.CLS_TOKEN,
        BLO_TOKEN: Optional[str] = Constants.BLO_WORD,
        lower: bool = True,
    ):
        """
        Args:
            name (str): name of vocabulary.
            data_file (Union[Path,str]): path of file containing training data to use for the vocabulary.
            filename (Union[Path,str]): path of vocabulary file to either load from, or save to.
            vocab_size (int, Optional): If set, sets cap on vocabulary size.
            max_length (int): maxiumum token length of a sequence.
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
        self.vocab: Any
        self.lower = lower

        self.BOS_TOKEN = BOS_TOKEN
        self.EOS_TOKEN = EOS_TOKEN
        self.UNK_TOKEN = UNK_TOKEN
        self.SEP_TOKEN = SEP_TOKEN
        self.PAD_TOKEN = PAD_TOKEN
        self.CLS_TOKEN = CLS_TOKEN
        self.BLO_TOKEN = BLO_TOKEN
        self.SPECIALS = {self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN}

    @deprecated
    def make(self, data_path: Union[Path, str]):
        raise NotImplementedError

    def fit(self, texts: Iterable[str]):
        """Trains tokenizer on dataset.

        Args:
            texts (Iterable[str]): Iterable of texts,
                e.g. ["hello world", "how are you?"]
        """
        raise NotImplementedError

    @deprecated
    def load(self):
        raise NotImplementedError

    @deprecated
    def save(self, filepath: Optional[Union[Path, str]] = None):
        if not filepath:
            filepath = self.filename
        raise NotImplementedError

    def to_tensor(self, text: str) -> torch.LongTensor:
        raise NotImplementedError

    def tokenize(text: str) -> List[str]:
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {
            "kwargs": {
                "name": self.name,
                "filename": self.filename,
                "vocab_size": self.vocab_size,
                "max_length": self.max_length,
                "vocab": self.vocab,
                "lower": self.lower,
                "BOS_TOKEN": self.BOS_TOKEN,
                "EOS_TOKEN": self.EOS_TOKEN,
                "UNK_TOKEN": self.UNK_TOKEN,
                "SEP_TOKEN": self.SEP_TOKEN,
                "PAD_TOKEN": self.PAD_TOKEN,
                "CLS_TOKEN": self.CLS_TOKEN,
                "BLO_TOKEN": self.BLO_TOKEN,
            }
        }

    def to_file(self, filepath: Path | str) -> None:
        """save vocabulary file.

        Args:
            filepath (Path | str): filepath to dump the JSON to.
        """
        if isinstance(filepath, str):
            filepath: Path = Path(filepath)
        dump = self.to_dict()
        with open(filepath, "w") as f:
            json.dump(dump, f)

    @classmethod
    def from_dict(cls, d: dict) -> Vocabulary:
        return Vocabulary(**d["kwargs"])

    @classmethod
    def from_file(cls, filepath: Path | str) -> Vocabulary:
        """Loads an encoder from path saved with save"""
        if isinstance(filepath, str):
            filepath: Path = Path(filepath)
        with open(filepath) as infile:
            obj = json.load(infile)
        return cls.from_dict(obj)

    def __repr__(self) -> str:
        return f"Vocabulary(vocab_size={self.vocab_size}, max_length={self.max_length}, lower={self.lower})"


class BPEVocabulary(Vocabulary):
    def __init__(self, name: str, pct_bpe: float, **kwargs):
        super().__init__(name, **kwargs)
        self.pct_bpe = pct_bpe
        self.vocab: BPEEncoder

    @deprecated
    def make(self, data_path: Union[Path, str]) -> None:
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
        if self.lower:
            dataset = [i.lower() for i in dataset]

        src_bpe.fit(dataset)
        self.vocab = src_bpe
        log.info("Finished training BPEEncoder.")

    def fit(self, texts: Iterable[str]):
        src_bpe = BPEEncoder(
            vocab_size=self.vocab_size,
            pct_bpe=self.pct_bpe,
            ngram_min=1,
            UNK=self.UNK_TOKEN,
            PAD=self.PAD_TOKEN,
            word_tokenizer=bpe_parse,
        )
        if self.lower:
            texts = [i.lower() for i in texts]

        src_bpe.fit(texts)
        self.vocab = src_bpe
        log.info("Finished training BPEEncoder.")

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

    @deprecated
    def load(self):
        self.vocab = BPEEncoder.load(self.filename)

    @deprecated
    def save(self, filepath: Optional[Union[Path, str]] = None):
        if not filepath:
            filepath = self.filename
        log.info(f"Saving vocabulary '{self.name}' to '{filepath}'...")
        self.vocab.save(filepath)

    @classmethod
    def from_file(cls, filepath: Path | str) -> BPEVocabulary:
        return super().from_file(filepath)

    @classmethod
    def from_dict(cls, d: dict) -> BPEVocabulary:
        this = cls(**d["kwargs"])
        this.vocab = BPEEncoder.from_dict(d["vocab"])
        return this

    def to_dict(self) -> dict:
        d = super().to_dict()
        del d["kwargs"]["vocab"]
        d["vocab"] = self.vocab.vocabs_to_dict()
        d["kwargs"]["pct_bpe"] = self.pct_bpe
        d["class"] = "BPEVocabulary"
        # log.info("dict", dict=d)
        return d

    def detokenize(
        self, tokens: List[List[int]], specials=True
    ) -> Iterable[str | list[str]]:
        # TODO clean up, this looks horrific
        out = []
        if isinstance(tokens[0], list):
            # nested list
            for item in tokens:
                if not specials:
                    item = [
                        [i for i in item if self.vocab.word_vocab[self.vocab.PAD] != i]
                    ]
                    out.append(next(self.vocab.inverse_transform(item)))
            return out
        else:
            if not specials:
                tokens = [
                    [i for i in item if self.vocab.word_vocab[self.vocab.PAD] != i]
                ]
                return next(self.vocab.inverse_transform(tokens))

    def tokenize(self, text: str) -> List[str]:
        return self.vocab.tokenize(text)

    def __repr__(self) -> str:
        return f"BPEVocabulary(vocab_size={self.vocab_size}, max_length={self.max_length}, lower={self.lower}, pct_bpe={self.pct_bpe})"


class SentencePieceVocabulary(Vocabulary):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vocab: spm.SentencePieceProcessor

    def fit(
        self,
        **kwargs,
    ):
        raise NotImplementedError("Train this with train_sentencepiece.py")

    @classmethod
    def from_file(
        cls, filepath: Path | str, max_length: int
    ) -> SentencePieceVocabulary:
        """Loads SentencePiece model from path.

        Args:
            filepath (Path | str): _description_

        Returns:
            SentencePieceVocabulary: _description_
        """
        filename = Path(filepath).name
        filepath = str(filepath)
        vocab = spm.SentencePieceProcessor(model_file=filepath)
        ID_BOS, ID_EOS, ID_PAD, ID_UNK = (
            vocab.bos_id(),
            vocab.eos_id(),
            vocab.pad_id(),
            vocab.unk_id(),
        )
        BOS, EOS, PAD, UNK = (
            vocab.IdToPiece(ID_BOS),
            vocab.IdToPiece(ID_EOS),
            vocab.IdToPiece(ID_PAD),
            vocab.IdToPiece(ID_UNK),
        )
        vocab_size = len(vocab)
        this = cls(
            name=filename,
            vocab_size=vocab_size,
            max_length=max_length,
            filename=filepath,
            BOS_TOKEN=BOS,
            EOS_TOKEN=EOS,
            PAD_TOKEN=PAD,
            UNK_TOKEN=UNK,
        )
        this.vocab = vocab
        return this

    def to_dict(self) -> dict:
        raise NotImplementedError

    def tokenize(self, text: str) -> List[str]:
        return self.vocab.encode(text, out_type=str)

    def detokenize(
        self, tokens: List[List[int]], specials=True
    ) -> Iterable[str | List[str]]:
        return self.vocab.decode(tokens)

    def to_tensor(self, text: str | List[str]) -> Iterable[List[int]]:
        if isinstance(text, str):
            text = [text]
        token_sequences = self.vocab.encode(text, out_type=int)
        for i, tokens in enumerate(token_sequences):
            tokens = tokens[: self.max_length - 2]
            tokens = [self.vocab.bos_id()] + tokens + [self.vocab.eos_id()]
            tokens = tokens + [self.vocab.pad_id()] * (self.max_length - len(tokens))
            token_sequences[i] = tokens
        return torch.LongTensor(token_sequences)


class WordVocabulary(Vocabulary):
    """White-space level tokenization, leveraging Moses."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vocab: onmt.Dict
        self.mt = MosesTokenizer(lang="en")
        self.md = MosesDetokenizer(lang="en")

    def fit(self, texts: Iterable[str]):
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
        for sentence in texts:
            for word in self.tokenize(sentence):
                vocab.add(word)

        original_size = vocab.size()

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
        )

        self.vocab = vocab.prune(self.vocab_size)
        log.info(
            "Created space-separated token dictionary of size %d (pruned from %d)"
            % (self.vocab.size(), original_size)
        )

    @deprecated
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

    @deprecated
    def load(self):
        if Path(self.filename).exists():
            # If given, load existing word dictionary.
            log.info(f"Reading {self.name} vocabulary from {self.filename}..")
            self.vocab.loadFile(self.filename)
            self.vocab_size = self.vocab.size()
            self.vocab.max_length = self.max_length
            self.lower = self.lower
            log.info(
                f"Loaded {self.vocab.size()} {self.name} tokens.",
                max_len=self.vocab.max_length,
            )
        else:
            raise FileNotFoundError(f"{self.filename} doesn't exist.")

    @deprecated
    def save(self, filepath: Optional[Union[Path, str]] = None):
        if not filepath:
            filepath = self.filename
        """Save vocabulary to file"""
        log.info(f"Saving vocabulary '{self.name}' to '{filepath}'...")
        self.vocab.writeFile(filepath)

    @classmethod
    def from_file(cls, filepath: Path | str) -> Vocabulary:
        return super().from_file(filepath)

    def to_dict(self) -> dict:
        d = super().to_dict()
        del d["kwargs"]["vocab"]
        d["vocab"] = {
            "idxToLabel": self.vocab.idxToLabel,
            "labelToIdx": self.vocab.labelToIdx,
            "frequencies": self.vocab.frequencies,
            "special": self.vocab.special,
        }
        d["class"] = "WordVocabulary"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> WordVocabulary:
        this = cls(**d["kwargs"])
        this.vocab = onmt.Dict(
            None, lower=d["kwargs"]["lower"], seq_len=d["kwargs"]["max_length"]
        )
        # loading ints from json
        idx2label = d["vocab"]["idxToLabel"]
        idx2label = {int(k): v for k, v in idx2label.items()}
        this.vocab.idxToLabel = idx2label

        this.vocab.labelToIdx = d["vocab"]["labelToIdx"]
        this.vocab.frequencies = d["vocab"]["frequencies"]
        this.vocab.special = d["vocab"]["special"]
        return this

    def to_tensor(self, text: str) -> torch.LongTensor:
        """Converts string to text to tensor of input.

        Args:
            text (str): Raw string of input.

        Returns:
            torch.LongTensor: List of indices corresponding to the token index from the vocab.
        """
        if isinstance(text, str):
            text: str
            return torch.LongTensor(
                self.vocab.convertToIdx(
                    self.tokenize(text),
                    unkWord=self.UNK_TOKEN,
                    padding=True,
                    bosWord=self.BOS_TOKEN,
                    eosWord=self.EOS_TOKEN,
                    paddingWord=self.PAD_TOKEN,
                )
            )
        else:
            # import numpy as np
            tokens = [
                self.vocab.convertToIdx(
                    self.tokenize(line),
                    unkWord=self.UNK_TOKEN,
                    padding=True,
                    bosWord=self.BOS_TOKEN,
                    eosWord=self.EOS_TOKEN,
                    paddingWord=self.PAD_TOKEN,
                )
                for line in text
            ]
            # tokens = np.array(tokens)
            return torch.LongTensor(tokens)

    def tokenize(self, text: str) -> List[str]:
        """Performs space level tokenization.
        Ensure that Moses Tokenization is used here."""
        return [word for word in self.mt.tokenize(text, return_str=True).split()]

    def detokenize(self, tokens: torch.LongTensor, specials=True) -> str:
        """Returns detokenized string.

        Args:
            tokens (torch.LongTensor): Tensor of integer values corresponding
            to indexes of tokens.
            specials (bool, optional): If unset, then removes special tokens from output. Defaults to True.

        Returns:
            str: plain text value.

        Example:
            text = "récompensée et toi"
            tokens = word_vocab.to_tensor(tokens)
            >>> tokens
            tensor([    2, 29907,    10,  7724,     3,     0,     0 ... ])
            >>> word_vocab.detokenize(tokens)
            "<s> récompensée et toi </s> <blank> <blank> ..."
            >>> word_vocab.detokenize(tokens, specials=False)
            "récompensée et toi"
        """
        labels = [self.vocab.idxToLabel[t.item()] for t in tokens]
        if not specials:
            labels = [t for t in labels if t not in self.SPECIALS]
        return self.md.detokenize(labels)

    def __repr__(self) -> str:
        return f"WordVocabulary(vocab_size={self.vocab_size}, max_length={self.max_length}, lower={self.lower})"
