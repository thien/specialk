from nltk.corpus import words
import unicodedata
from tqdm import tqdm
import argparse
import sys

sys.path.append('../base/core')

from utils import get_len

def load_args():
    desc = """
    sanitise.py

    deals with convering the mose style datasets into data that can be interpreted by the neural models (for machine translation or style transfer).
    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-source_a', required=True,
                        help="Path to the training source data for locale A")
    parser.add_argument('-a_label', required=True,
                        help="label for locale A")
    parser.add_argument('-source_b', required=True,
                        help="Path to the training source data for locale B")
    parser.add_argument('-b_label', required=True,
                        help="label for locale B")
    parser.add_argument('-cutoff', default=0.6,
                        help="Cutoff ratio for filtering.")
    parser.add_argument('-verbose', default=True, help="Output logs or not.")

    opt = parser.parse_args()
    assert opt.a_label.lower() == "en" or opt.b_label.lower() == "en"
    return opt

def sanitise(en_seq):
    """
    Replaces characters.
    """
    return en_seq.replace('“', '"').replace('”','"').replace('’',"'").replace("·", "-").replace("– ", " ").replace("\xad", "").replace("‘", "'").replace("…", "...")

def en_filter(seq, cutoff):
    """
    Filters out non-ascii characters and determines similarity. If it's not similar (often the case for
    foreign languages, it'll return False. True otherwise.)
    """
    if len(seq) < 3:
        return False
    y = unicodedata.normalize('NFKD',seq).encode('ascii', 'ignore').decode('ascii')
    return sum([a == b for(a,b) in zip(seq,y)])/len(seq) >= cutoff

if __name__ == "__main__":
    # load args and dataset
    opt = load_args()

    m = tqdm if opt.verbose else iter

    with open(opt.source_a) as a, open(opt.source_b) as b:
        before = [(sanitise(x.strip()), sanitise(y.strip())) for (x,y) in tqdm(zip(a,b), desc="Loading Files", total=get_len(opt.source_a))]

    # determine the index where the english sequence is
    fi = 0 if opt.a_label.lower() == "en" else 1

    filt_desc = "Filtering Sequence Pairs"
    bad_i = {i for i in tqdm(range(len(before)), desc=filt_desc) if not en_filter(before[i][fi], opt.cutoff)}
    # keep sequences with pound signs.
    bad_i = {i for i in tqdm(bad_i) if "£" not in before[i][fi]}

    with open(opt.source_a, "w") as a, open(opt.source_b, "w") as b:
        for i in tqdm(range(len(before)), desc="Writing Files"):
            if i in bad_i:
                continue
            x, y = before[i]
            a.write(x + "\n")
            b.write(y + "\n")
