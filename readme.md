# Scalable Text-Based Style-Transfer

![Style Transfer from Popular Press to Quality](doc/images/preview.png)

[![wakatime](https://wakatime.com/badge/user/4251d8c6-f363-4bc2-92de-31e3923c596a/project/41ffd15c-a534-4e52-9250-6e308379d529.svg)](https://wakatime.com/badge/user/4251d8c6-f363-4bc2-92de-31e3923c596a/project/41ffd15c-a534-4e52-9250-6e308379d529)

This repository contains the implementation of what used to be my summer thesis. This has evolved over time. 

## Requirements

This project uses [`uv`](https://docs.astral.sh/uv/). After installing `uv` and cloning this repo, `cd` into it and run:

    uv sync
    make download-deps

That's it!

This has been tested to run on both Linux and macOS Sequoia and above, and has not been tested on Windows.

## Running

### Training

To train different models, see `specialk/trainers/*.py` for the different training use cases.

### Generating

To translate with a trained model, see [doc/generation.md](doc/generation.md).

## Datasets

To make life easier, I've set up a one-command auto running program that'll deal with downloading all the necessary machine-translation datasets needed to make your model. You'll need `wget` however.

    cd datasets
    ./init_enfr_dataset.sh
    ./political_data.sh

That being said, you'll have to compose your own newspaper dataset, since I'm near certain that releasing such a dataset is not allowed by the newspapers for a variety of reasons, ranging from ethical to legal. In the root directory there is a zip called newspapers. run the jupyter notebook called `downloader_manual.ipynb` to get the article dataset (you'll need to submit your login for The Times in `times.json` first).

If `times.json` isn't created use the following format:

    {
        "action" : "login",
        "username": "TIMES_USERNAME",
        "password": "PASSWORD",
        "s" : 1,
        "rememberMe" : "on"
    }

After running the notebook run `convert_express.sh` and it'll move it to the base directory.

## Running

Once the datasets are downloaded, `cd scripts/scripts/prod` and run the following:

    ./train_nmt_models.sh
    ./train_pol_st_models.sh
    ./build_pub_corpus.sh
    ./train_pub_st_models.sh
    ./train_pub_naturalness_models.sh

It'll probably take quite a while to train the models.

## Development

### Testing

Run the lightweight suite with:

    make tests-light

Run the full suite with:

    make tests

Benchmarks run on their own, serially, because they saturate the GPU:

    make tests-bench

You can also call `uv run pytest` directly. `pytest-watch` works as well.
