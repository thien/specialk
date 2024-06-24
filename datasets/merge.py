"""
Merge scripts to join multiple MOSE style corpus datasets together.
"""

import argparse
import os
import subprocess

from tqdm import tqdm


def get_len(filepath):
    """
    Reads number of lines in the mose corpus without using python
    to deal with it. This is some order of magnitude faster!
    """
    command = "wc -l " + filepath
    process = subprocess.run(command.split(" "), stdout=subprocess.PIPE)
    return int(process.stdout.decode("utf-8").split(" ")[0])


def load_args():
    parser = argparse.ArgumentParser(description="train.py")
    # data options

    parser.add_argument("-left", nargs="+", required=True, action="append")
    parser.add_argument("-right", nargs="+", required=True, action="append")
    parser.add_argument("-left_out", required=True, type=str)
    parser.add_argument("-right_out", required=True, type=str)

    return parser.parse_args()


args = load_args()
# python3 merge.py -left machine_translation/europarl/europarl.en machine_translation/global_voices/globalvoices.en  machine_translation/hansards/hansards.en -right machine_translation/europarl/europarl.fr machine_translation/global_voices/globalvoices.fr machine_translation/hansards/hansards.fr -left_out machine_translation/corpus_enfr.en -right_out machine_translation/corpus_enfr.en

dataset = []

for ent in zip(args.left[0], args.right[0]):
    fpl, fpr = ent
    length = get_len(fpl)
    with open(fpl) as l, open(fpr) as r:
        for pair in tqdm(zip(l, r), total=length):
            dataset.append(pair)

# check for bad pairs
bads = []
for i in tqdm(range(len(dataset))):
    pair = dataset[i]
    # check if the pair consists of content in both left and right
    left, right = [x.strip() for x in pair]
    if len(left) < 1 or len(right) < 1:
        bads.append(i)
    else:
        dataset[i] = (left, right)
# inspect integrity of bad sequences (seems fine)

# for i in bads:
#     pair = dataset[i]
#     left, right = [x.strip() for x in pair]
#     print(">", dataset[i-1][0],":-", dataset[i-1][1])
#     print(">>", left, ":-", right)
#     print(">", dataset[i+1][0],":-", dataset[i+1][1])
#     print()
#     break

# time to write to file
bads = set(bads)
with open(args.left_out, "w") as lf, open(args.right_out, "w") as rf:
    for i in tqdm(range(len(dataset)), desc="Writing"):
        if i in bads:
            continue
        pair = dataset[i]
        left, right = pair
        lf.write(left + "\n")
        rf.write(right + "\n")
