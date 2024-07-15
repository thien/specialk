import pytest

from specialk.metrics import (
    BLEU,
    LexicalMetrics,
    Meteor,
    Polarity,
)


@pytest.fixture
def lexical():
    return LexicalMetrics()


@pytest.fixture
def polarity():
    return Polarity()


@pytest.fixture
def bleu():
    return BLEU()


@pytest.fixture
def meteor():
    return Meteor()


def test_meteor(meteor):
    test_case = [
        (["hello world"], ["who are you"], 0),
        (["hello world"], ["hello world"], 0.9375),
        (["hello"], ["hello world"], 0.2631578947368421),
        (["hello world", "dog"], ["hello world", "cat"], 0.46875),
    ]
    for pred, actual, expected_score in test_case:
        assert meteor.compute(pred, actual) == expected_score


def test_bleu(bleu):
    test_case = [
        (
            ["hello world"],
            ["who are you"],
            {"bleu1": 1, "bleu2": 0, "bleu3": 0, "bleu4": 0},
        ),
        (
            ["hello world"],
            ["hello world"],
            {"bleu1": 1, "bleu2": 0, "bleu3": 0, "bleu4": 0},
        ),
        (["hello"], ["hello world"], {"bleu1": 1, "bleu2": 0, "bleu3": 0, "bleu4": 0}),
        (
            ["hello world", "dog"],
            ["hello world", "cat"],
            {"bleu1": 1, "bleu2": 0, "bleu3": 0, "bleu4": 0},
        ),
    ]
    for pred, actual, expected_score in test_case:
        assert bleu.compute(pred, actual) == expected_score


def test_syllables(lexical):
    raise NotImplementedError


def test_lexical_lex_match_1(lexical):
    raise NotImplementedError


def test_lexical_lex_match_2(lexical):
    raise NotImplementedError


def test_lexical_basic_stats(lexical):
    raise NotImplementedError
