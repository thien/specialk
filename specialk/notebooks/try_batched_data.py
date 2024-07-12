import pandas as pd
from specialk.core.utils import load_dataset
from pathlib import Path
import datasets
from datasets import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader

parent_dir = Path("/Users/t/Projects/specialk/datasets/machine_translation/")

from specialk.models.tokenizer import BPEVocabulary, BPEEncoder, WordVocabulary

word_vocab = WordVocabulary.from_file(
    "/Users/t/Projects/specialk/assets/tokenizer/fr_en_word_moses"
)


def tokenize(example):
    # perform tokenization at this stage.
    example["source"] = word_vocab.to_tensor(example["source"])
    example["target"] = word_vocab.to_tensor(example["target"])
    return example


batched_data = Dataset.load_from_disk(
    "/Users/t/Projects/specialk/datasets/machine_translation/huggingface/corpus_enfr",
    keep_in_memory=True,
)


def collator(batch):
    src = [i["source"] for i in batch]
    tgt = [i["target"] for i in batch]
    return {"source": word_vocab.to_tensor(src), "target": word_vocab.to_tensor(tgt)}


torchdata = DataLoader(
    batched_data.with_format("torch"), batch_size=3, collate_fn=collator, num_workers=2
)

j = 100
k = 0
for i in tqdm(torchdata):
    k += 1
    if k == j:
        break
    continue
