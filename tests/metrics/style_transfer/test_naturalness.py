from specialk.metrics import (
    Naturalness,
)
import pytest


@pytest.fixture
def naturalness():
    return Naturalness()


def test_naturalness(naturalness):
    raise NotImplementedError
