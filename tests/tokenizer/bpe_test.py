from specialk.core.bpe import Encoder
from tqdm import tqdm

sequences = []
with open("../../datasets/multi30k/train.en") as f:
    for line in f:
        sequences.append(line.strip())

ref = [x.split() for x in sequences]
ref_len = [len(x) for x in ref]
print("REF:", max(ref_len))


def parse(x):
    return x.split()


enc = Encoder(
    4096, ngram_min=1, ngram_max=2, pct_bpe=0.8, silent=True, word_tokenizer=parse
)
enc.fit(sequences)

base = enc.vocabs_to_dict()
duplicate_keys = []
for key in base["byte_pairs"]:
    if key in base["words"]:
        duplicate_keys.append(key)
if len(duplicate_keys) > 0:
    print("got duplicates:")
    print(duplicate_keys)
else:
    print("NO DUPLICATES! :)")

keybase = {**base["words"], **base["byte_pairs"]}


inv_map = {v: k for k, v in keybase.items()}

for i in range(0, 10):
    print(i, inv_map[i])

sequences = [f for f in enc.transform(tqdm(sequences))]

lengths = [len(x) for x in sequences]
print(max(lengths))

# print(base)
