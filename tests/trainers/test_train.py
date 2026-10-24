import pandas as pd
import pytest
from datasets import Dataset

from specialk.core.constants import RNN, SOURCE, TARGET, TRANSFORMER
from specialk.trainers import train as train_module
from specialk.trainers.train import (
    build_parser,
    load_datasets,
    resolve_hyperparameters,
)


@pytest.mark.lightweight
def test_parser_defaults():
    args = build_parser().parse_args([])

    assert args.model == TRANSFORMER
    assert args.dataset == "dummy"
    assert args.accelerator == "auto"
    assert args.prod is False
    assert args.limit_train_batches == 1.0
    assert args.limit_val_batches == 1.0
    assert args.val_check_interval == 1.0
    assert args.max_epochs is None


@pytest.mark.lightweight
def test_parser_prod_and_smoke_flags():
    args = build_parser().parse_args(
        [
            "--dataset",
            "mt",
            "--prod",
            "--accelerator",
            "mps",
            "--limit-train-batches",
            "0.01",
            "--limit-val-batches",
            "0.01",
        ]
    )

    assert args.dataset == "mt"
    assert args.prod is True
    assert args.accelerator == "mps"
    assert args.limit_train_batches == 0.01


@pytest.mark.lightweight
def test_limit_args_parse_as_batch_counts():
    args = build_parser().parse_args(
        [
            "--limit-train-batches",
            "1",
            "--limit-val-batches",
            "2",
            "--val-check-interval",
            "0.5",
        ]
    )

    # Lightning treats ints as batch counts and floats as epoch fractions.
    assert args.limit_train_batches == 1
    assert isinstance(args.limit_train_batches, int)
    assert args.limit_val_batches == 2
    assert isinstance(args.limit_val_batches, int)
    assert args.val_check_interval == 0.5


@pytest.mark.lightweight
def test_whole_floats_keep_fraction_semantics():
    args = build_parser().parse_args(
        [
            "--limit-train-batches",
            "1.0",
            "--limit-val-batches",
            "1.0",
            "--val-check-interval",
            "1.0",
        ]
    )

    # 1.0 means "the whole epoch", not "one batch".
    assert isinstance(args.limit_train_batches, float)
    assert args.limit_train_batches == 1.0
    assert isinstance(args.limit_val_batches, float)
    assert isinstance(args.val_check_interval, float)


@pytest.mark.lightweight
def test_prod_can_be_disabled_for_shims():
    args = build_parser().parse_args(["--prod", "--no-prod"])

    assert args.prod is False


@pytest.mark.lightweight
def test_prod_transformer_config():
    hyperparameters = resolve_hyperparameters(model=TRANSFORMER, prod=True)

    assert hyperparameters["batch_size"] == 64
    assert hyperparameters["max_seq_len"] == 100
    assert hyperparameters["n_epochs"] == 6
    assert hyperparameters["model_config"]["dim_model"] == 512
    assert hyperparameters["model_config"]["num_encoder_layers"] == 6


@pytest.mark.lightweight
def test_prod_rnn_config_uses_larger_batch():
    hyperparameters = resolve_hyperparameters(model=RNN, prod=True)

    assert hyperparameters["batch_size"] == 192
    assert hyperparameters["max_seq_len"] == 75
    assert hyperparameters["model_config"]["name"] == "lstm"


@pytest.mark.lightweight
def test_smoke_config_is_small():
    hyperparameters = resolve_hyperparameters(model=TRANSFORMER, prod=False)

    assert hyperparameters["batch_size"] == 192
    assert hyperparameters["max_seq_len"] == 30
    assert hyperparameters["model_config"]["dim_model"] == 128
    assert hyperparameters["model_config"]["name"] == "transformer_smol"


@pytest.mark.lightweight
def test_explicit_values_override_defaults():
    hyperparameters = resolve_hyperparameters(
        model=TRANSFORMER,
        prod=True,
        batch_size=8,
        max_seq_len=12,
        n_epochs=1,
    )

    assert hyperparameters["batch_size"] == 8
    assert hyperparameters["max_seq_len"] == 12
    assert hyperparameters["n_epochs"] == 1


@pytest.mark.lightweight
def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        resolve_hyperparameters(model="cnn")


def _write_parquet(path, source, target, n_rows=2):
    pd.DataFrame(
        {
            SOURCE: [source] * n_rows,
            TARGET: [target] * n_rows,
            "source_lang": [source] * n_rows,
            "target_lang": [target] * n_rows,
        }
    ).to_parquet(path)


@pytest.mark.lightweight
def test_load_datasets_dummy_reads_given_paths(tmp_path):
    train_path = tmp_path / "train.parquet"
    valid_path = tmp_path / "valid.parquet"
    _write_parquet(train_path, "de", "en", n_rows=4)
    _write_parquet(valid_path, "de", "en", n_rows=2)

    bundle = load_datasets("dummy", train_path=train_path, valid_path=valid_path)

    assert bundle.task_name == "nmt_model_dummy"
    assert bundle.train.num_rows == 4
    assert bundle.valid.num_rows == 2
    assert bundle.train_path == train_path
    assert bundle.valid_path == valid_path


@pytest.mark.lightweight
def test_load_datasets_dummy_back_translation_swaps_columns(tmp_path):
    train_path = tmp_path / "train.parquet"
    valid_path = tmp_path / "valid.parquet"
    _write_parquet(train_path, "de", "en", n_rows=1)
    _write_parquet(valid_path, "de", "en", n_rows=1)

    bundle = load_datasets(
        "dummy",
        train_path=train_path,
        valid_path=valid_path,
        back_translation=True,
    )

    assert bundle.train[0][SOURCE] == "en"
    assert bundle.train[0][TARGET] == "de"
    assert bundle.train[0]["source_lang"] == "en"
    assert bundle.train[0]["target_lang"] == "de"


@pytest.mark.lightweight
def test_publications_rejects_back_translation():
    with pytest.raises(ValueError, match="not supported"):
        load_datasets("publications", back_translation=True)


@pytest.mark.lightweight
def test_mt_back_translation_swaps_languages(monkeypatch):
    seen = {}

    def fake_train(src_lang, tgt_lang, num_proc=8):
        seen["train"] = (src_lang, tgt_lang)
        return Dataset.from_dict({SOURCE: ["x"], TARGET: ["y"]})

    def fake_validation(src_lang, tgt_lang, num_proc=8):
        seen["valid"] = (src_lang, tgt_lang)
        return Dataset.from_dict({SOURCE: ["x"], TARGET: ["y"]})

    monkeypatch.setattr(train_module, "load_mt_training_dataset", fake_train)
    monkeypatch.setattr(train_module, "load_mt_validation_dataset", fake_validation)

    bundle = train_module.load_datasets("mt", back_translation=True, n_workers=2)

    assert seen["train"] == ("en", "fr")
    assert seen["valid"] == ("en", "fr")
    assert bundle.task_name == "nmt_model"
