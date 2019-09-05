## Requirements

You'll need `python3`, `pytorch`, `spacy`, `numpy`, `pyemd`, `bayesian-optimisation`, `rouge`, and `pyTelegramBotAPI`. You can install them via `pip3` (but you'll want to make use of CUDA so you might want to look into that too).

To make life easier I've added a `requirements.txt` that'll allow you to install everything necessary (after installing `python3.6+`):

        pip install -e .

## Datasets

To make life easier, I've set up a one-command auto running program that'll deal with downloading all the necessary machine-translation datasets needed to make your model. You'll need `wget` however.

    cd datasets
    ./master_enfr.sh
    ./political_data.sh

That being said, you'll have to compose your own newspaper dataset, since I'm near certain that releasing such a dataset is not allowed by the newspapers for a variety of reasons, ranging from ethical to legal. 

## Running

Once the datasets are downloaded, `cd base/scripts` and run the following:

    cd prod
    ./train_nmt_models.sh
    ./train_pol_st_models.sh
    ./build_pub_corpus.sh
    ./train_pub_st_models.sh
    ./train_pub_naturalness_models.sh
