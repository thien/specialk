from typing import List, Tuple

import numpy as np
import pytest

from specialk.metrics import ROUGE, LexicalMetrics, Meteor, Polarity, SacreBLEU


@pytest.fixture
def lexical():
    return LexicalMetrics()


@pytest.fixture
def polarity():
    return Polarity()


@pytest.fixture
def rouge():
    return ROUGE()


@pytest.fixture
def meteor():
    return Meteor()


@pytest.fixture
def sacrebleu():
    return SacreBLEU()


@pytest.fixture
def pos_sequence_1(lexical) -> List[Tuple[str, str]]:
    sequence = "It's uncear what Teresa May is planning."
    pos_tokens = lexical.pos(sequence)
    return pos_tokens


@pytest.fixture
def pos_sequence_2(lexical) -> List[Tuple[str, str]]:
    sequence = "he was responsible for all for."
    pos_tokens = lexical.pos(sequence)
    return pos_tokens

@pytest.mark.lightweight
def test_meteor(meteor):
    test_case = [
        (["hello world"], ["who are you"], 0),
        (["hello world"], ["hello world"], 0.9375),
        (["hello"], ["hello world"], 0.2631578947368421),
        (["hello world", "dog"], ["hello world", "cat"], 0.46875),
    ]
    for pred, actual, expected_score in test_case:
        assert meteor.compute(pred, actual) == expected_score

@pytest.mark.lightweight
def test_rouge(rouge):
    rouge: ROUGE
    test_case = [
        (["this is a test"], ["this is a test"], 1.0),
        (
            ["this is a test", "hello world"],
            [["this is a test"], ["hello world hi"]],
            0.9,
        ),
    ]
    for pred, actual, score in test_case:
        pred_score = rouge.compute(pred, actual)
        assert np.isclose(pred_score, score, rtol=1e-02, atol=1e-03)

@pytest.mark.lightweight
def test_sacrebleu(sacrebleu):
    test_case = [
        (["this is a test"], ["this is a test"], 100.0),
        (
            ["this is a test", "hello world"],
            [["this is a test"], ["hello world hi"]],
            84.64,
        ),
    ]
    for pred, actual, score in test_case:
        pred_score = sacrebleu.compute(pred, actual)
        assert np.isclose(pred_score, score, rtol=1e-02, atol=1e-03)


@pytest.mark.lightweight
def test_syllables(lexical):
    assert lexical.syllables("word") > 0


@pytest.mark.lightweight
def test_lexical_lex_match_1(lexical, pos_sequence_1):
    scores = lexical.lex_match_1(pos_sequence_1)
    assert len(scores) > 0

@pytest.mark.lightweight
def test_lexical_lex_match_2(lexical, pos_sequence_2):
    scores = lexical.lex_match_2(pos_sequence_2)
    assert len(scores) > 0


@pytest.mark.lightweight
def test_lexical_basic_stats(lexical):
    article = "The quick brown fox jumped over the lazy dog. He moved and jumped."
    lexical.basic_stats(article)
