import pytest

from specialk.metrics import EarthMoverDistance

# The constants depend on the GloVe snapshot and on float32 embedding math, so
# allow solver/accumulation differences at the 1e-6 level.


@pytest.fixture(scope="module")
def emd():
    return EarthMoverDistance()


@pytest.mark.heavyweight
def test_emd_perfect(emd):
    assert emd.compute(
        prediction="hello world", references="hello world"
    ) == pytest.approx(0.0)


@pytest.mark.heavyweight
def test_emd_great(emd):
    assert (
        emd.compute(prediction="hello world", references="hi world")
        == pytest.approx(2.2670567114660742, rel=1e-5)
    )


@pytest.mark.heavyweight
def test_emd_poor(emd):
    assert (
        emd.compute(prediction="bye world", references="hello world")
        == pytest.approx(2.8166197714643477, rel=1e-5)
    )


@pytest.mark.heavyweight
def test_emd_horror(emd):
    assert (
        emd.compute(prediction="chicken wings", references="hello world")
        == pytest.approx(8.018476176153182, rel=1e-5)
    )
