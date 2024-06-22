"""
ENSURE THAT THE CONSTANTS ARE
CAPITALISED.
"""

PAD = 0
UNK = 1
SOS = 2
EOS = 3
BLO = 4

PAD_WORD = "<blank>"
UNK_WORD = "<unk>"
SOS_WORD = "<s>"
EOS_WORD = "</s>"
BLO_WORD = "<p>"  # paragraph block


def get_tokens():
    g = globals()
    k = list({i for i in g if i[-5:] == "_WORD"})
    kc = [i[:-5] for i in k]
    k = sorted(kc, key=lambda x: g[x])
    return [g[i + "_WORD"] for i in k]


if __name__ == "__main__":
    print(get_tokens())
