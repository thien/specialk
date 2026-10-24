"""Fresh-process checkpoint round trips for both HuggingFace-backed modules.

Each case saves a trained model in one interpreter and loads it in another.
That is deliberate: Lightning/PEFT can keep enough state alive in-process to
hide a checkpoint that never captured the adapter.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("checkpoint_worker.py")
DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def _run(phase, arch, use_peft, device, checkpoint, reference, result=None):
    command = [
        sys.executable,
        str(WORKER),
        phase,
        "--arch",
        arch,
        "--device",
        device,
        "--checkpoint",
        str(checkpoint),
        "--reference",
        str(reference),
    ]
    if use_peft:
        command.append("--peft")
    if result is not None:
        command.extend(["--result", str(result)])

    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{phase} worker failed for {arch} peft={use_peft} device={device}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


@pytest.mark.heavyweight
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("use_peft", [False, True], ids=["no-peft", "peft"])
@pytest.mark.parametrize("arch", ["bert", "marian"])
def test_checkpoint_roundtrip_fresh_process(tmp_path, arch, use_peft, device):
    checkpoint = tmp_path / "model.ckpt"
    reference = tmp_path / "reference.pt"
    result_path = tmp_path / "result.json"

    _run(
        "save",
        arch,
        use_peft,
        device,
        checkpoint,
        reference,
    )
    _run(
        "load",
        arch,
        use_peft,
        device,
        checkpoint,
        reference,
        result_path,
    )

    result = json.loads(result_path.read_text())

    assert result["peft"] is use_peft
    assert result["n_weights"] > 0
    assert result["weight_keys_match"], "saved and loaded weight keys differ"
    assert result["weights_max_diff"] <= 1e-7
    # Same inputs, same device, fresh interpreter.  MPS kernels are not
    # bitwise deterministic across processes, so allow a small relative error.
    tolerance = 1e-6 + 1e-5 * result["logits_abs_max"]
    assert result["logits_max_diff"] <= tolerance
