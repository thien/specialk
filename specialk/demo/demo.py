import argparse
from tqdm import tqdm
import sys

sys.path.append("../")
from lib.TransformerModel import TransformerModel as transformer
from train import load_args
import torch
from core.bpe import Encoder as BPE
from preprocess import reclip

from googletrans import Translator as GTranslator
from core.dataset import TranslationDataset, collate_fn, paired_collate_fn
from lib.transformer.Translator import Translator

googt = GTranslator()
import subprocess
import os
from nltk.tokenize import sent_tokenize
import torch.nn.functional as F
import torch.nn as nn
from mosestokenizer import MosesDetokenizer

detokeniser = MosesDetokenizer("en")
import textwrap
from termcolor import colored, cprint

import sys

sys.path.append("/home/t/Data/Files/Github/msc_project")
from Downloader import Downloader


pub_colours = {"popular": "red", "quality": "blue"}
classes = {
    "thesun": "popular",
    "mirror": "popular",
    "theguardian": "quality",
    "thetimes": "quality",
    "bbc": "quality",
}


def flip_class(inp):
    if inp.lower() == "popular":
        return "quality"
    return "popular"


do_quality_to_popular = (
    True  # if true: do quality to popular, else do popular to quality
)


def extract_quotes(line, split_sentences=True):
    quotes, norms, sequence = [], [], []
    quote_indexes = set([i for i, x in enumerate(line) if x == '"'])

    if len(quote_indexes) < 1:
        if split_sentences:
            sequences = sent_tokenize(line)
        else:
            sequences = [line]
        return {
            "sequences": sequences,
            "quotes": [],
            "norms": [i for i in range(len(sequences))],
            "equivalent": " ".join(sequences) == line,
        }

    i, on_quote = 0, False
    current_sequence = ""
    sequences = []
    while i != len(line):
        char = line[i]
        if i in quote_indexes:
            if not on_quote:
                on_quote = True
                if i != 0:
                    # split sequence if it consists of multiple sentences
                    sentences = sent_tokenize(current_sequence)
                    for s in sentences:
                        sequences.append(s)
                        norms.append(len(sequences) - 1)
                    current_sequence = ""
            else:
                on_quote = False
                sequences.append(current_sequence + char)
                i += 1
                quotes.append(len(sequences) - 1)
                current_sequence = ""
                continue

        current_sequence += char
        i += 1

    if len(current_sequence) > 0:
        if on_quote:
            sequences.append(current_sequence)
            quotes.append(len(sequences) - 1)
        else:
            sentences = sent_tokenize(current_sequence)
            for s in sentences:
                sequences.append(s)
                norms.append(len(sequences) - 1)

    sequences = [x.strip() for x in sequences]

    return {
        "sequences": sequences,
        "quotes": quotes,
        "norms": norms,
        "equivalent": " ".join(sequences) == line,
    }


def parse_article(paragraphs):
    # seperate quotes and lines in paragraph
    parsed_p = [extract_quotes(x) for x in paragraphs]

    sequences = []
    norms = []
    quotes = []
    paragraphs = []

    for p in parsed_p:
        inc = len(sequences)
        sequences += p["sequences"]
        norms += [inc + i for i in p["norms"]]
        quotes += [inc + i for i in p["quotes"]]
        paragraphs.append(inc)

    return {
        "sequences": sequences,
        "quotes": quotes,
        "norms": norms,
        "paragraphs": paragraphs,
    }


def dl_handler(args):
    """
    single process handler for downloading an article.
    It calls the outlet.get_article() function.
    returns the newspaper file, and potential URLs found in the article.
    """
    outlet, url = args
    article = dl.outlets[outlet].get_article(url)
    if article:
        # we don't want the article object because we can't serialise that
        # and it's unnecessary weight.
        urls = dl.outlets[outlet].scrape_articles_in_urls(article)
        article = dl.outlets[outlet].parse(article)
        if article:
            return (outlet, url, article, urls)
    return (outlet, url, None, set())


def save_raw_article_out(sequences, en_filename):
    with open(en_filename, "w") as f:
        for line in sequences:
            f.write(line + "\n")


def return_tokenise_file(filepath, lang="en"):
    # needs to be absolute filepath
    cmd = "./tokenise_file.sh" if lang == "en" else "./tokenise_file_fr.sh"
    l = subprocess.check_output(cmd.split(), shell=True, cwd=cwd)

    with open(filepath + ".atok") as f:
        lines = [i.strip() for i in f]
    return lines


def prettyprint(contents, overwritenorms=None):
    sequences = contents["sequences"]

    quot_idx = set(contents["quotes"])
    para_idx = set(contents["paragraphs"])
    norm_idx = set(contents["norms"])

    output = []

    onorms_i = 0

    with MosesDetokenizer("en") as detokenize:
        for i in tqdm(range(len(sequences)), desc="Formatting"):
            if i in para_idx and i != 0:
                output.append("\n\n")
            if i in norm_idx:
                if not overwritenorms:
                    seq = detokenize(sequences[i].split())
                else:
                    seq = detokenize(overwritenorms[onorms_i].split())
                    onorms_i += 1
            elif i in quot_idx:
                seq = detokenize(sequences[i].split())
            output.append(seq)

    return " ".join(output)


def diff_filter(line, type="src"):
    ob = "[-" if type != "src" else "{+"
    cb = "-]" if type != "src" else "+}"

    out = ""
    i = 0
    to_write = True
    while i < len(line):
        char = line[i]
        if char == ob[-1] and line[i - 1] == ob[0]:
            to_write = False
        elif char == cb[-1] and line[i - 1] == cb[0]:
            to_write = True
        if to_write:
            out += char
        i += 1
    bug = "{}" if type == "src" else "[]"
    return out.replace(bug, "")


def printdiff(seqs, srclabel, tgtlabel, block_width=50, spacing_block_len=2):
    pad = " ".join(["" for _ in range(block_width)])
    src_col = pub_colours[srclabel]
    tgt_col = pub_colours[tgtlabel]
    src_before, src_after = [x for x in colored(" ", color=src_col).split()]
    tgt_before, tgt_after = [x for x in colored(" ", color=tgt_col).split()]

    for i in range(len(seqs)):
        if seqs[i] == "\n":
            print()
            continue
        if seqs[i][0:2] == "@@":
            continue
        srcline, tgtline = (
            diff_filter(seqs[i].strip()),
            diff_filter(seqs[i].strip(), "tgt"),
        )
        left = textwrap.wrap(srcline, width=block_width)
        right = textwrap.wrap(tgtline, width=block_width)

        largest_min = min([len(left), len(right)])
        largest_max = max([len(left), len(right)])

        left_spacing = []
        for i in range(len(left)):
            left_line = left[i]

            opened = False
            j = 0
            left_space_diff = 0
            while j < len(left_line) - 1:
                j += 1
                if left_line[j : j + 1] == "[-":
                    left_space_diff += 2
                    if opened == False:
                        opened = True
                        j += 1
                    else:
                        left_line = left_line + src_after
                        break
                elif left_line[j : j + 1] == "-]":
                    left_space_diff += 2
                    if opened == True:
                        opened = False
                        j += 1
                    else:
                        left_line = src_before + left_line
                        break

            left_line_before = (
                len(left_line.replace("[-", "").replace("-]", "")) - spacing_block_len
            )

            left_line = (
                left_line.replace("[-", " " + src_before).replace("-]", src_after)
                + src_after
            )
            left_line = " ".join(left_line.split())
            left_line_before = (
                len(left_line.replace(src_before, "").replace(src_after, ""))
                - spacing_block_len
            )
            spacing = " ".join("" for _ in range(block_width - left_line_before))
            left[i] = left_line
            left_spacing.append(spacing)

        for i in range(len(right)):
            right_line = right[i]
            opened = False
            j = 0
            right_space_diff = 0
            while j < len(right_line) + 1:
                if right_line[j : j + 2] == "{+":
                    if opened == False:
                        opened = True
                if right_line[j : j + 2] == "+}":
                    if opened == True:
                        opened = False
                        j += 1
                    else:
                        right_line = tgt_before + right_line
                        break
                j += 1

            right_line = (
                right_line.replace("{+", tgt_before).replace("+}", tgt_after)
                + tgt_after
            )
            right[i] = right_line

        left_is_largest = len(left) > len(right)

        for i in range(largest_min):
            print(left[i], left_spacing[i], right[i])

        for i in range(largest_min, largest_max):
            if left_is_largest:
                print(left[i])
            else:
                whole_pad = pad + "".join(" " for _ in range(spacing_block_len + 2))
                print(whole_pad + right[i])


def do_back_translation(test_loader, tl, encmodel, decoder):
    generated_outputs = []
    count = 1
    tl.max_token_seq_len = max_seq_length
    for i in tqdm(test_loader, desc="Generating Style Responses"):
        a, b = i
        result, _ = tl.translate_batch(a, b)
        lines = encmodel.translate_decode_bpe(result, decoder)
        for j in lines:
            generated_outputs.append(j)

    return generated_outputs


def do_forward_translation(n):
    # translate it into french
    french_sequences = [
        googt.translate(x, src="en", dest="fr").text
        for x in tqdm(n, desc="Forward Translation")
    ]

    with open(fr_filename, "w") as f:
        for line in french_sequences:
            f.write(line.lower() + "\n")


def handle_git_diff(srclabel, tgtlabel):
    try:
        l = subprocess.check_output("./diff.sh", shell=True, cwd=cwd)
    except:
        pass

    # load git diff
    with open(os.path.join(cwd, "diff.text")) as f:
        dt = [i for i in f][5:]

    # print response
    printdiff(dt, srclabel, tgtlabel)


max_seq_length = 100
cwd = os.getcwd()
os.getcwd()
en_filename = os.path.join(cwd, "test_article.txt")
fr_filename = os.path.join(cwd, "test_article.txt.fr")
dump_src_filepath = os.path.join(cwd, "src_article_dump.txt")
dump_tgt_filepath = os.path.join(cwd, "tgt_article_dump.txt")
test_txt = fr_filename

basedir = "/media/data/Files/Github/msc_project_model/base/"

# vocabulary_path = os.path.join(basedir,"models/fren_bpe_gm_mk2_articles_mk1/gm_mk2_rebase_quality.pt")
# encoder_path = os.path.join(basedir,"models/fren_bpe_gold_master_mk2/encoder_epoch_3.chkpt")
# decoder_path = "models/fren_bpe_gm_mk2_articles_mk1/decoder_popular_epoch_3.chkpt" if do_quality_to_popular else"models/fren_bpe_gm_mk2_articles_mk1/decoder_quality_epoch_4.chkpt"
# decoder_path = os.path.join(basedir,decoder_path)

# args = load_args()
# args.data = vocabulary_path
# args.model = "transformer"
# args.epochs = "1"
# opt = args.parse_args(["-data", args.data, "-model", args.model, "-epochs", args.epochs])
# opt.batch_size = 1
# opt.checkpoint_encoder = encoder_path
# opt.checkpoint_decoder = decoder_path
# opt.override_max_token_seq_len = False
# opt.cuda = True
# opt.beam_size = 5
# opt.n_best = 1

# encmodel = transformer(opt)

# # load vocabulary
# encmodel.load_dataset()

# encmodel.load(opt.checkpoint_encoder, opt.checkpoint_decoder)

# translator = Translator(encmodel.opt, new=False)
# translator.model = encmodel.model
# translator.model.eval()


def load_transformer_model(decoder_path):
    args = load_args()
    args.data = os.path.join(
        basedir, "models/fren_bpe_gm_mk2_articles_mk1/gm_mk2_rebase_quality.pt"
    )
    args.model = "transformer"
    args.epochs = "1"
    opt = args.parse_args(
        ["-data", args.data, "-model", args.model, "-epochs", args.epochs]
    )
    opt.batch_size = 1
    opt.checkpoint_encoder = os.path.join(
        basedir, "models/fren_bpe_gold_master_mk2/encoder_epoch_3.chkpt"
    )
    opt.checkpoint_decoder = os.path.join(basedir, decoder_path)
    opt.override_max_token_seq_len = False
    opt.cuda = False
    opt.beam_size = 5
    opt.n_best = 1

    encmodel = transformer(opt)

    # load vocabulary
    encmodel.load_dataset()

    encmodel.load(opt.checkpoint_encoder, opt.checkpoint_decoder)

    translator = Translator(encmodel.opt, new=False)
    translator.model = encmodel.model
    translator.model.eval()
    return translator, encmodel


popular_translator, popular_model = load_transformer_model(
    "models/fren_bpe_gm_mk2_articles_mk1/decoder_popular_epoch_3.chkpt"
)
quality_translator, quality_model = load_transformer_model(
    "models/fren_bpe_gm_mk2_articles_mk1/decoder_quality_epoch_4.chkpt"
)

print("Loaded Transformer")

dl = Downloader()
sites = [
    "https://www.thesun.co.uk/news/politics/",
    "https://www.mirror.co.uk/news/politics/",
    "https://www.theguardian.com/politics/",
    "https://www.bbc.co.uk/news/politics",
]
dl.init_outlets(sites)


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------


# testurl="https://www.theguardian.com/politics/2019/sep/01/get-ready-for-brexit-government-launches-information-blitz"
# testurl="https://www.theguardian.com/world/2019/sep/02/hong-kong-protests-calls-grow-to-give-citizens-right-to-live-and-work-in-uk"
testurl = "https://www.theguardian.com/politics/2019/sep/07/boris-johnson-could-be-jailed-for-refusing-to-seek-brexit-delay"
testurl = (
    "https://www.theguardian.com/world/2019/sep/07/hurricane-dorian-survivors-bahamas"
)
testurl = "https://www.theguardian.com/world/2019/sep/09/us-removed-covert-source-in-russia-due-to-safety-concerns-under-trump"


def handle_style_transfer(testurl):
    print("LOADING:", testurl)
    pub_label = testurl.split("/")[2]
    for key in classes.keys():
        if key in pub_label:
            pub_label = key
            break
    if pub_label not in classes:
        print("ERR: cant find srcgroup in url.")
        exit()

    srclabel = classes[pub_label]
    tgtlabel = flip_class(srclabel)

    dl.queue = {(pub_label, testurl)}

    results = dl.batch(dl_handler, dl.queue)

    sequences = [
        x.replace("”", '"')
        .replace("“", '"')
        .replace("’", "'")
        .replace("'", "'")
        .lower()
        for x in results[0][2]["content"]
        if x
    ]

    save_raw_article_out(sequences, en_filename)

    paragraphs = return_tokenise_file(en_filename)

    contents = parse_article(paragraphs)
    norms = [contents["sequences"][i] for i in contents["norms"]]

    do_forward_translation(norms)

    fr_paragraphs = return_tokenise_file(os.path.join(cwd, fr_filename), "fr")

    # generate back-translation results
    vocab = os.path.join(
        basedir, "models/fren_bpe_gm_mk2_articles_mk1/gm_mk2_rebase_quality.pt"
    )
    translator = quality_translator if tgtlabel == "quality" else popular_translator
    encmodel = quality_model if tgtlabel == "quality" else popular_model
    test_loader, max_token_seq_len, is_bpe, decoder = encmodel.load_testdata(
        test_txt, vocab
    )

    generated_outputs = do_back_translation(test_loader, translator, encmodel, decoder)
    # save outputs to file
    output = prettyprint(contents)
    gen_out = prettyprint(contents, generated_outputs)

    with open(dump_src_filepath, "w") as f:
        for line in output.split("\n"):
            f.write(line + "\n")
    with open(dump_tgt_filepath, "w") as f:
        for line in gen_out.split("\n"):
            f.write(line + "\n")

    print("\n\n")
    # print visualisation of git differences
    handle_git_diff(srclabel, tgtlabel)


while True:
    print("Note: don't put quotes!")
    testurl = input("Provide URL (or type in `e` to exit): ")
    if testurl.lower() == "e":
        break
    print("\n\n")
    handle_style_transfer(testurl)
    print("\n\n")
