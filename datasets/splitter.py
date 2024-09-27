import argparse
import os
import random

from tqdm import tqdm

desc = """
splitter.py

deals with loading moses style aligned datasets into training data, validation data, and test data. 

This is naturally not needed if the dataset is already appropiately split.
"""

parser = argparse.ArgumentParser(description=desc)
parser.add_argument(
    "-source_a", required=True, help="Path to the training source data for locale A"
)
parser.add_argument("-a_label", required=True, help="label for locale A")
parser.add_argument(
    "-source_b", required=True, help="Path to the training source data for locale B"
)
parser.add_argument("-b_label", required=True, help="label for locale B")
parser.add_argument(
    "-seed", default=0, help="Randomiser seed used to shuffle the dataset."
)
parser.add_argument(
    "-ratio",
    default="80:15:5",
    help="Ratio to determine the dataset split. Input should be in the form of train:val:test. See default example as a guide.",
)
parser.add_argument(
    "-save_location",
    help="Location to save the output files. defaults to the same location of the source file.",
)
parser.add_argument(
    "-verbose",
    action="store_true",
    default=False,
    help="Determines whether to print outputs.",
)

opt = parser.parse_args()

# apply seed
random.seed(opt.seed)
# setup ratio splits.
ratios = [float(i) for i in opt.ratio.split(":")]
ratio_total = sum(ratios)
assert len(ratios) == 3
# make sure that the files actually exist.
assert os.path.isfile(opt.source_a)
assert os.path.isfile(opt.source_b)
# setup file save location.
save_dir = os.path.abspath(os.path.join(opt.source_a, os.pardir))
filename_group = os.path.normpath(opt.source_a).split(os.sep)[-1].split(".")[0]

# read number of lines in the mose corpus without using python
# to deal with it.
import subprocess

command = "wc -l " + opt.source_a
process = subprocess.run(command.split(" "), stdout=subprocess.PIPE)
num_ents = int(process.stdout.decode("utf-8").split(" ")[0])

# load corpus files.
dataset = []
with open(opt.source_a) as a, open(opt.source_b) as b:
    for lang1, lang2 in tqdm(
        zip(a, b), total=num_ents, desc="read", disable=not opt.verbose
    ):
        dataset.append((lang1, lang2))

# shuffle dataset indexes.
indexes = [i for i in range(len(dataset))]
random.shuffle(indexes)


# normalise ratios
def f_round(f):
    # python3's rounding function is broken.
    decimals = f % 1
    whole = int(f - decimals)
    return whole + 1 if decimals > 0.5 else whole


# calculate index points to start splitting the dataset.
ratios = [0] + [i / ratio_total for i in ratios]
ratios = [f_round(i * len(dataset)) for i in ratios]
# setup indexes to iterate by.
for i in range(len(ratios)):
    portion = ratios[i]
    if i > 0:
        portion += ratios[i - 1]
    ratios[i] = portion
# rearrange dataset.
indexes = [indexes[ratios[j - 1] : ratios[j]] for j in range(1, len(ratios))]
dataset = [[dataset[i] for i in j] for j in indexes]

datatypes = ["train", "val", "test"]

# save dataset splits.
p = "w"
for i in tqdm(range(len(datatypes)), desc="write", disable=not opt.verbose):
    group_name = datatypes[i]
    group_dataset = dataset[i]
    filename = filename_group + "." + group_name
    filepath = os.path.join(save_dir, filename)
    a_filename = filepath + "." + opt.a_label
    b_filename = filepath + "." + opt.b_label
    with open(a_filename, p) as s, open(b_filename, p) as t:
        for s_txt, t_txt in tqdm(
            group_dataset, desc=group_name, disable=not opt.verbose
        ):
            s.write(s_txt)
            t.write(t_txt)
