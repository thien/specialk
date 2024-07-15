import pytest

from specialk.metrics import (
    Naturalness,
)


@pytest.fixture
def naturalness():
    return Naturalness()


def test_naturalness(naturalness):
    raise NotImplementedError
