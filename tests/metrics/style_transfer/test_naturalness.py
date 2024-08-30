import pytest

from specialk.metrics import Naturalness


@pytest.fixture
def naturalness():
    return Naturalness("political")


def test_naturalness(naturalness):
    test_natural_tweets = [
        "let 'federalize all gun crimes.",
        "assault weapons make no sense except in the armed services.",
        "he just came out of the communist closet.",
        "he's a true patriot!",
    ]

    avg_score = naturalness.compute(test_natural_tweets)
    assert avg_score > 0.85

    test_unnatural_tweets = [
        "aaaaaaa asdfasdf a8e4fa weif asdfjo",
        "the the the the the",
        "asdf 123 123 123 2 3 4 1 5 2" "hello",
    ]

    avg_score = naturalness.compute(test_unnatural_tweets)
    assert avg_score < 0.5
