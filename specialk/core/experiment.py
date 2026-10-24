"""Experiment logging shared by the training entry points.

W&B is the default logger. TensorBoard stays available with
``--logger tensorboard`` for the old run directories.
"""

import argparse

from lightning.pytorch.loggers import Logger, TensorBoardLogger

from specialk.core.constants import LOGGING_DIR, PROJECT_DIR


def add_logger_args(
    parser: argparse.ArgumentParser, default_group: str
) -> argparse.ArgumentParser:
    """Add the logger flags every trainer shares."""
    parser.add_argument(
        "--logger",
        choices=["wandb", "tensorboard"],
        default="wandb",
        help="experiment logger",
    )
    parser.add_argument(
        "--wandb-project",
        default="specialk",
        help="W&B project to log to",
    )
    parser.add_argument(
        "--wandb-group",
        default=default_group,
        help="W&B group for the run",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="logger run name (default: the experiment name)",
    )
    parser.add_argument(
        "--wandb-run-id",
        default=None,
        help="continue an existing W&B run when resuming",
    )
    return parser


def init_logger(
    args: argparse.Namespace, name: str, params: dict
) -> Logger:
    """Build the logger for a run and record its settings."""
    if args.logger == "wandb":
        from lightning.pytorch.loggers import WandbLogger

        run_name: str | None = args.run_name
        if run_name is None and not args.wandb_run_id:
            run_name = name.replace("/", "-")
        extra = {}
        if args.wandb_run_id:
            extra = {"id": args.wandb_run_id, "resume": "allow"}
        return WandbLogger(
            project=args.wandb_project,
            group=args.wandb_group,
            name=run_name,
            save_dir=PROJECT_DIR,
            config=params,
            **extra,
        )

    logger = TensorBoardLogger(LOGGING_DIR, name=name)
    logger.log_hyperparams(params=params)
    return logger
