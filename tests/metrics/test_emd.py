from specialk.metrics import (
    EarthMoverDistance,
)

EMD = EarthMoverDistance()
# not using a fixture here since it's a lot slower to run.
# import pytest
# @pytest.fixture
# def EMD():
#     return EarthMoverDistance()


def test_emd_perfect():
    assert EMD.compute(prediction="hello world", references="hello world") == 0.0


def test_emd_great():
    assert (
        EMD.compute(prediction="hello world", references="hi world")
        == 2.2670567114660742
    )


def test_emd_poor():
    assert (
        EMD.compute(prediction="bye world", references="hello world")
        == 2.8166197714643477
    )


def test_emd_horror():
    assert (
        EMD.compute(prediction="chicken wings", references="hello world")
        == 8.018476176153182
    )
