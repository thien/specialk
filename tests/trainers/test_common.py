import pandas as pd
import pytest
from datasets import Dataset, DatasetDict

from specialk.core.constants import SOURCE, TARGET
from specialk.trainers import common
from specialk.trainers.common import (
    _single_split,
    _strip_index_column,
    invert_df_columns,
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
def test_load_publications_datasets_normalises_columns(monkeypatch):
    train = Dataset.from_dict(
        {"fr_text": ["bonjour"], "text": ["hello"], "label": [1]}
    )
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
