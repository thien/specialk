import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from specialk.models.utils.lid import FastTextLID


@pytest.fixture
def mock_fasttext_model():
    # should a dummy version of fasttext.FastText
    mock_model = Mock()
    mock_model.f.multilinePredict.return_value = (
        [["__label__en"], ["__label__fr"]],
        [[0.9], [0.8]],
    )
    return mock_model


@pytest.fixture
def mock_fasttext_lid(mock_fasttext_model):
    with patch("fasttext.load_model", return_value=mock_fasttext_model):
        with patch("huggingface_hub.hf_hub_download", return_value="mock_path"):
            return FastTextLID()


@pytest.fixture
def sample_data():
    return pd.Series(["Hello world", "Bonjour monde"])


def test_process_batch(mock_fasttext_lid, mock_fasttext_model):
    batch = ["Hello", "World"]
    labels, probs = mock_fasttext_lid.process_batch(
        batch, k=2, threshold=0.0, on_unicode_error="strict"
    )

    mock_fasttext_model.f.multilinePredict.assert_called_once_with(
        batch, 2, 0.0, "strict"
    )
    assert isinstance(labels, np.ndarray)
    assert isinstance(probs, np.ndarray)
    assert labels.shape == (2,)
    assert probs.shape == (2,)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_lid_scores(mock_fasttext_lid, sample_data, batch_size):
    with patch("multiprocessing.Pool") as mock_pool:
        mock_pool.return_value.__enter__.return_value.imap.return_value = [
            (
                np.array(
                    [["__label__en", "__label__fr"], ["__label__fr", "__label__en"]]
                ),
                np.array([[0.9, 0.1], [0.8, 0.2]]),
            )
        ]

        labels, scores = mock_fasttext_lid.lid_scores(
            sample_data, batch_size=batch_size
        )

        assert isinstance(labels, np.ndarray)
        assert isinstance(scores, np.ndarray)
        assert labels.shape == (2, 2)
        assert scores.shape == (2, 2)
        assert labels[:, 0].tolist() == ["__label__en", "__label__fr"]
        assert scores[:, 0].tolist() == [0.9, 0.8]


def test_lid_scores_empty_input(mock_fasttext_lid):
    empty_series = pd.Series([])
    labels, scores = mock_fasttext_lid.lid_scores(empty_series)
    assert len(labels) == 0
    assert len(scores) == 0


def test_init_worker(mock_fasttext_lid):
    with patch("fasttext.load_model") as mock_load_model:
        FastTextLID._init_worker("mock_path")
        mock_load_model.assert_called_once_with("mock_path")


@pytest.fixture
def language_code():
    return FastTextLID().language


def test_enum_creation(language_code):
    assert language_code.ENG_Latn.value == "eng_Latn"
    assert language_code.FRA_Latn.value == "fra_Latn"
    assert language_code.SPA_Latn.value == "spa_Latn"


def test_from_string_with_prefix(language_code):
    assert language_code.from_string("__label__eng_Latn") == language_code.ENG_Latn
    assert language_code.from_string("__label__fra_Latn") == language_code.FRA_Latn


def test_from_string_without_prefix(language_code):
    assert language_code.from_string("eng_Latn") == language_code.ENG_Latn
    assert language_code.from_string("fra_Latn") == language_code.FRA_Latn


def test_from_string_invalid(language_code):
    with pytest.raises(ValueError):
        language_code.from_string("invalid_code")


def test_numpy_array_of_enums(language_code):
    enum_array = np.array(
        [language_code.ENG_Latn, language_code.FRA_Latn, language_code.SPA_Latn]
    )
    assert enum_array.dtype == object
    assert all(isinstance(item, language_code) for item in enum_array)


def test_numpy_array_of_strings():
    value_array = np.array(["eng_Latn", "fra_Latn", "spa_Latn"])
    assert value_array.dtype == "<U8"


def test_string_array_to_enum_array(language_code):
    def string_array_to_enum_array(arr):
        vec_func = np.vectorize(language_code.from_string)
        return vec_func(arr)

    test_array = np.array(["__label__eng_Latn", "fra_Latn", "__label__spa_Latn"])
    enum_array = string_array_to_enum_array(test_array)

    assert all(isinstance(item, language_code) for item in enum_array)
    assert enum_array[0] == language_code.ENG_Latn
    assert enum_array[1] == language_code.FRA_Latn
    assert enum_array[2] == language_code.SPA_Latn


def test_enum_value_consistency(language_code):
    for member in language_code:
        assert "__label__" not in member.value
        assert language_code.from_string(f"__label__{member.value}") == member
        assert language_code.from_string(member.value) == member
