"""
ENSURE THAT THE CONSTANTS ARE CAPITALISED.
"""

from pathlib import Path
from typing import List

import specialk

PAD = 0
UNK = 1
SOS = 2
EOS = 3
BLO = 4
SEP = 5

PAD_WORD = "<blank>"
UNK_WORD = "<unk>"
SOS_WORD = "<s>"
EOS_WORD = "</s>"
BLO_WORD = "<p>"  # paragraph block
SEP_WORD = "<sep>"
CLS_TOKEN = "<cls>"
URL_TOKEN = "<URL>"

SOURCE = "source"
TARGET = "target"

RNN = "rnn"
TRANSFORMER = "transformer"


def get_tokens() -> List[str]:
    g = globals()
    k = list({i for i in g if i[-5:] == "_WORD"})
    kc = [i[:-5] for i in k]
    k = sorted(kc, key=lambda x: g[x])
    return [g[i + "_WORD"] for i in k]


PROJECT_DIR: Path = Path(specialk.__file__).parent.parent

# for tensorboard; logging
LOGGING_DIR: Path = PROJECT_DIR / "tb_logs"
LOGGING_PERF_NAME: str = "perf_logs"
