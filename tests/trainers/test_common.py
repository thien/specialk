from pathlib import Path

import pandas as pd
import pytest

from datasets import Dataset, DatasetDict
from specialk.core.constants import SOURCE, TARGET
from specialk.trainers import common
from specialk.trainers.common import (
    _single_split,
    _strip_index_column,
    cache_filename,
    invert_df_columns,
    load_style_datasets,
)

STYLE_FIXTURE = (
    Path(__file__).parents[1] / "test_files" / "datasets" / "en_fr_labels.parquet"
)


@pytest.mark.lightweight
def test_single_split_returns_named_split():
    dataset = Dataset.from_dict({"x": [1, 2]})
    splits = DatasetDict({"train": dataset, "eval": dataset})

    assert _single_split(splits, "eval").num_rows == 2


@pytest.mark.lightweight
def test_single_split_accepts_plain_dataset():
    dataset = Dataset.from_dict({"x": [1]})

    assert _single_split(dataset, "train") is dataset


@pytest.mark.lightweight
def test_single_split_rejects_missing_split():
    dataset = Dataset.from_dict({"x": [1]})

    with pytest.raises(ValueError, match="available splits"):
        _single_split(DatasetDict({"train": dataset}), "eval")


@pytest.mark.lightweight
def test_strip_index_column_is_a_no_op_without_index():
    dataset = Dataset.from_dict({"x": [1]})

    assert _strip_index_column(dataset).column_names == ["x"]


@pytest.mark.lightweight
def test_strip_index_column_removes_pandas_index():
    dataset = Dataset.from_dict({"__index_level_0__": [0, 1], "x": [1, 2]})

    assert _strip_index_column(dataset).column_names == ["x"]


@pytest.mark.lightweight
def test_invert_df_columns_swaps_source_and_target():
    df = pd.DataFrame(
        {
            SOURCE: ["de"],
            TARGET: ["en"],
            "source_lang": ["de"],
            "target_lang": ["en"],
        }
    )

    inverted = invert_df_columns(df)

    assert inverted[SOURCE].tolist() == ["en"]
    assert inverted[TARGET].tolist() == ["de"]
    assert inverted["source_lang"].tolist() == ["en"]
    assert inverted["target_lang"].tolist() == ["de"]


@pytest.mark.lightweight
def test_cache_filename_changes_with_tokenization_settings():
    base = cache_filename("train.parquet", tokenizer="spm", max_seq_len=100)

    assert base != cache_filename("train.parquet", tokenizer="bpe", max_seq_len=100)
    assert base != cache_filename("train.parquet", tokenizer="spm", max_seq_len=75)
    assert base != cache_filename(
        "train.parquet", tokenizer="spm", max_seq_len=100, back_translation=True
    )


@pytest.mark.lightweight
def test_cache_filename_distinguishes_same_named_paths():
    first = cache_filename("/data/v1/train.parquet", tokenizer="spm", max_seq_len=100)
    second = cache_filename("/data/v2/train.parquet", tokenizer="spm", max_seq_len=100)

    assert first != second


@pytest.mark.lightweight
def test_load_publications_datasets_normalises_columns(monkeypatch):
    train = Dataset.from_dict({"fr_text": ["bonjour"], "text": ["hello"], "label": [1]})
    valid = Dataset.from_dict({"fr_text": ["salut"], "text": ["hi"], "label": [0]})

    monkeypatch.setattr(
        common,
        "load_dataset",
        lambda *args, **kwargs: DatasetDict({"train": train, "eval": valid}),
    )

    train_dataset, valid_dataset = common.load_publications_datasets(num_proc=1)

    assert set(train_dataset.column_names) == {SOURCE, TARGET, "label"}
    assert set(valid_dataset.column_names) == {SOURCE, TARGET, "label"}
    assert train_dataset[0][SOURCE] == "bonjour"
    assert train_dataset[0][TARGET] == "hello"


@pytest.mark.lightweight
def test_load_style_datasets_renames_and_encodes_the_fixture():
    train, eval_split = load_style_datasets(
        "political", path=STYLE_FIXTURE, source_column="text_fr", num_proc=1
    )

    assert train.column_names == [SOURCE, TARGET, "label"]
    assert eval_split.column_names == [SOURCE, TARGET, "label"]
    # The fixture has one split, which is both train and eval.
    assert train.num_rows == eval_split.num_rows == 10
    assert train[0][SOURCE].startswith("Merci")
    assert train[0][TARGET].startswith("thank you")
    assert train.features["label"].dtype == "int64"
    # political: republican=0, democratic=1
    assert train[0]["label"] == 1


@pytest.mark.lightweight
def test_load_style_datasets_filters_by_target_label():
    democratic, _ = load_style_datasets(
        "political", target_label=1, path=STYLE_FIXTURE, source_column="text_fr"
    )
    republican, _ = load_style_datasets(
        "political", target_label=0, path=STYLE_FIXTURE, source_column="text_fr"
    )

    assert democratic.num_rows == 10
    assert set(democratic["label"]) == {1}
    assert republican.num_rows == 0


@pytest.mark.lightweight
def test_load_style_datasets_class_encodes_publication_labels():
    train, _ = load_style_datasets(
        "publications", path=STYLE_FIXTURE, source_column="text_fr"
    )

    assert train.features["label"].dtype == "int64"


@pytest.mark.lightweight
def test_load_style_datasets_rejects_missing_source_column():
    with pytest.raises(ValueError, match="available columns"):
        load_style_datasets("political", path=STYLE_FIXTURE, source_column="text_de")


@pytest.mark.lightweight
def test_load_style_datasets_reads_a_separate_eval_file(tmp_path):
    columns = {"text_fr": ["bonjour"], "text": ["hello"], "label": [1]}
    train_path = tmp_path / "train.parquet"
    eval_path = tmp_path / "dev.parquet"
    pd.DataFrame(columns).to_parquet(train_path, index=False)
    pd.DataFrame({"text_fr": ["salut"], "text": ["hi"], "label": [0]}).to_parquet(
        eval_path, index=False
    )

    train, eval_split = load_style_datasets(
        "political",
        path=train_path,
        eval_path=eval_path,
        source_column="text_fr",
    )

    assert train.num_rows == 1
    assert eval_split.num_rows == 1
    assert train[0][SOURCE] == "bonjour"
    assert eval_split[0][SOURCE] == "salut"
    assert eval_split[0]["label"] == 0
