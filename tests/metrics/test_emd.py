from specialk.metrics import (
    EarthMoverDistance,
)
import pytest


@pytest.fixture
def emd():
    return EarthMoverDistance()


def test_emd(emd):
    raise emd.compute(["hello world"], ["hello world"], 1) 
