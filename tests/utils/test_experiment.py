"""Tests for the shared experiment logger."""

import argparse

from specialk.core import experiment


def logger_args(*argv: str, group: str = "classifiers") -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    experiment.add_logger_args(parser, group)
    return parser.parse_args(list(argv))


def test_logger_defaults_to_wandb():
    args = logger_args()
    assert args.logger == "wandb"
    assert args.wandb_project == "specialk"
    assert args.wandb_group == "classifiers"
    assert args.run_name is None


def test_tensorboard_logger_uses_the_logging_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "LOGGING_DIR", tmp_path / "tb_logs")
    args = logger_args("--logger", "tensorboard")

    logger = experiment.init_logger(args, "dataset/model", {"lr": 0.1})

    assert logger.name == "dataset/model"
    assert str(tmp_path) in logger.log_dir


def test_wandb_logger_passes_the_run_settings(monkeypatch):
    captured = {}

    class FakeWandbLogger:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("lightning.pytorch.loggers.WandbLogger", FakeWandbLogger)
    args = logger_args()

    experiment.init_logger(args, "dataset/model", {"lr": 0.1})

    assert captured["group"] == "classifiers"
    assert captured["name"] == "dataset-model"
    assert captured["config"] == {"lr": 0.1}


def test_wandb_resume_keeps_the_existing_name(monkeypatch):
    captured = {}

    class FakeWandbLogger:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("lightning.pytorch.loggers.WandbLogger", FakeWandbLogger)
    args = logger_args("--wandb-run-id", "abc123")

    experiment.init_logger(args, "dataset/model", {})

    assert captured["id"] == "abc123"
    assert captured["resume"] == "allow"
    assert captured["name"] is None
