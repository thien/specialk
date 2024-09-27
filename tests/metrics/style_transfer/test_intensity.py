import pytest

from specialk.metrics import Intensity


@pytest.fixture
def intensity():
    return Intensity("political")


@pytest.mark.heavyweight
def test_intensity(intensity):
    test_democrat_tweets = [
        "lets 'federalize all gun crimes.",
        "assault weapons make no sense except in the armed services.",
    ]

    avg_score = intensity.compute(test_democrat_tweets)
    assert avg_score < 0.5

    test_republican_tweets = [
        "he just came out of the communist closet.",
        "he's a true patriot!",
    ]

    avg_score = intensity.compute(test_republican_tweets)
    assert avg_score > 0.1
