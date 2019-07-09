## Requirements

You'll need `python3`, `pytorch`, `spacy`, `numpy`, `pyemd`, `bayesian-optimisation`, `rouge`, and `pyTelegramBotAPI`. You can install them via `pip3` (but you'll want to make use of CUDA so you might want to look into that too).

To make life easier I've added a `requirements.txt` that'll allow you to install everything necessary (after installing `python3.6+`):

        pip install -e .

## Datasets

To make life easier, I've set up a one-command auto running program that'll deal with downloading all the necessary datasets needed. You'll need `wget` however.

That being said, you'll have to compose your own newspaper dataset, since I'm near certain that releasing such a dataset is not allowed by the newspapers for a variety of reasons, ranging from ethical to legal.