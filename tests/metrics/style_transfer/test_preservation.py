import pytest

from specialk.core.constants import PROJECT_DIR
from specialk.metrics import Preservation


@pytest.fixture
def preservation():
    return Preservation(
        dir_cache=PROJECT_DIR / "cache",
        dir_w2v=PROJECT_DIR / "cache" / "embeddings",
        path_lexicon=PROJECT_DIR / "cache" / "style_lexicons",
    )


@pytest.mark.heavyweight
def test_preservation(preservation):
    text_close = ("the fox was sleeping", "the fox slept")
    text_far = ("the fox was sleeping", "chicken wings are gerat")
    score_close = preservation.compute(text_close[0], text_close[1])
    score_far = preservation.compute(text_far[0], text_far[1])
    assert score_close < score_far
