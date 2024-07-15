import pytest

from specialk.metrics import (
    Preservation,
)


@pytest.fixture
def preservation():
    return Preservation()


def test_preservation(preservation):
    raise NotImplementedError
