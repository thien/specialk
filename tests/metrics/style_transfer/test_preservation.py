from specialk.metrics import (
    Preservation,
)
import pytest


@pytest.fixture
def preservation():
    return Preservation()


def test_preservation(preservation):
    raise NotImplementedError
