import pytest

from specialk.metrics import (
    Intensity,
)


@pytest.fixture
def intensity():
    return Intensity()


def test_intensity(intensity):
    raise NotImplementedError
