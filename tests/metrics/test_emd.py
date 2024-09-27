import pytest

from specialk.metrics import EarthMoverDistance


@pytest.fixture(scope="module")
def emd():
    return EarthMoverDistance()


@pytest.mark.heavyweight
def test_emd_perfect(emd):
    assert emd.compute(prediction="hello world", references="hello world") == 0.0


@pytest.mark.heavyweight
def test_emd_great(emd):
    assert (
        emd.compute(prediction="hello world", references="hi world")
        == 2.2670567114660742
    )


@pytest.mark.heavyweight
def test_emd_poor(emd):
    assert (
        emd.compute(prediction="bye world", references="hello world")
        == 2.8166197714643477
    )


@pytest.mark.heavyweight
def test_emd_horror(emd):
    assert (
        emd.compute(prediction="chicken wings", references="hello world")
        == 8.018476176153182
    )
