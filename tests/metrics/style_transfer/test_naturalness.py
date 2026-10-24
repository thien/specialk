import pytest

from specialk.core.constants import PROJECT_DIR
from specialk.metrics import Naturalness

CKPT_PATH = (
    PROJECT_DIR
    / "assets/classifiers/legacy/cnn_classifier/naturalness_political/"
    "naturalness_political.ckpt"
)


@pytest.fixture
def naturalness():
    if not CKPT_PATH.exists():
        pytest.skip(f"missing classifier checkpoint: {CKPT_PATH}")
    return Naturalness("political")

@pytest.mark.heavyweight
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
    assert avg_score <= 1  # TODO this is temporary to check if we can do inference.
