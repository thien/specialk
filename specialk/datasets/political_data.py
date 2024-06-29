# TODO: Address all TODOs and remove all explanatory comments
"""TODO: Add a description here."""

import os
from pathlib import Path
from typing import List

import datasets

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@inproceedings{style_transfer_acl18,
    title={Style Transfer Through Back-Translation},
    author={Prabhumoye, Shrimai and Tsvetkov, Yulia and Salakhutdinov, Ruslan and Black, Alan W},
    year={2018},
    booktitle={Proc. ACL}
    }
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "main": "http://tts.speech.cs.cmu.edu/style_models/political_data.tar",
}


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class PoliticalDataset(datasets.GeneratorBasedBuilder):
    """TODO: Political Tweets Dataset between Democratic and Republican."""

    VERSION = datasets.Version("1.0.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="main",
            version=VERSION,
            description="Political Tweets Dataset.",
        ),
    ]

    DEFAULT_CONFIG_NAME = "main"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        if (
            self.config.name == "main"
        ):  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.Value("string"),
                    # These are the features of your dataset like images, labels ...
                }
            )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            supervised_keys=("text", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive

        url = _URLS[self.config.name]
        archive = dl_manager.download(url)
        data_dir = dl_manager.iter_archive(archive)

        for file in data_dir:
            print(file)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": [
                        os.path.join(data_dir, "republican_only.train.en"),
                        os.path.join(data_dir, "democratic_only.train.en"),
                    ],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": [os.path.join(data_dir, "classtrain.txt")],
                    "split": "train_class",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": [
                        os.path.join(data_dir, "republican_only.dev.en"),
                        os.path.join(data_dir, "democratic_only.dev.en"),
                    ],
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": [
                        os.path.join(data_dir, "republican_only.test.en"),
                        os.path.join(data_dir, "democratic_only.test.en"),
                    ],
                    "split": "test",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepaths: List[str], split: str):
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        for filepath in filepaths:
            filename = Path(filepath).name
            label = filename.split(".")[0].split("_")[0]
            with open(filepath, encoding="utf-8") as f:
                for key, row in enumerate(f):
                    text = row.strip()
                    if split != "test":
                        # label only exists in train/eval files.
                        text = text.split()
                        label, text = text[0], text[1:]
                        text = " ".join(text)

                    yield (
                        key,
                        {"text": text, "label": label},
                    )
