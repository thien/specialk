from specialk.metrics import (
    Intensity,
)
import pytest


@pytest.fixture
def intensity():
    return Intensity()


def test_intensity(intensity):
    raise NotImplementedError
