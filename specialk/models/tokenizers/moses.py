"""Moses pre-tokenisation for the 2019 checkpoints."""

from sacremoses import MosesTokenizer


class MosesPreTokenizer:
    """Match ``tokenizer.perl -a -no-escape -l <lang>`` plus lowercasing.

    This is the modern Python port used by the archived 2019 models. It is
    applied before BPE encoding; the BPE vocabulary is in the model's
    ``vocab.pt`` / ``tokenizer_*.json`` files.
    """

    def __init__(self, lang: str):
        self.tokenizer = MosesTokenizer(lang=lang)

    def __call__(self, text: str) -> str:
        return self.tokenizer.tokenize(
            text,
            aggressive_dash_splits=True,
            escape=False,
            return_str=True,
        ).lower()
