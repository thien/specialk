from specialk.metrics import Meteor
import pytest


@pytest.fixture
def meteor():
    return Meteor()

def test_meteor_cases(meteor):
    test_case = [
        (["hello world"], ["who are you"], 0),
        (["hello world"], ["hello world"], 0.9375),
        (["hello"], ["hello world"], 0.2631578947368421),
        (["hello world", "dog"], ["hello world", "cat"], 0.46875),
    ]
    for pred, actual, expected_score in test_case:
        assert meteor.compute(pred, actual) == expected_score


