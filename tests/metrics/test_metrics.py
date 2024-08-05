from typing import List, Tuple

import pytest

from specialk.metrics import BLEU, LexicalMetrics, Meteor, Polarity, SacreBLEU


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


def test_meteor(meteor):
    test_case = [
        (["hello world"], ["who are you"], 0),
        (["hello world"], ["hello world"], 0.9375),
        (["hello"], ["hello world"], 0.2631578947368421),
        (["hello world", "dog"], ["hello world", "cat"], 0.46875),
    ]
    for pred, actual, expected_score in test_case:
        assert meteor.compute(pred, actual) == expected_score


# def test_bleu(bleu):
#     test_case = [
#         (
#             ["hello world"],
#             ["who are you"],
#             {"bleu1": 1, "bleu2": 0, "bleu3": 0, "bleu4": 0},
#         ),
#         (
#             ["hello world"],
#             ["hello world"],
#             {"bleu1": 1, "bleu2": 0, "bleu3": 0, "bleu4": 0},
#         ),
#         (["hello"], ["hello world"], {"bleu1": 1, "bleu2": 0, "bleu3": 0, "bleu4": 0}),
#         (
#             ["hello world", "dog"],
#             ["hello world", "cat"],
#             {"bleu1": 1, "bleu2": 0, "bleu3": 0, "bleu4": 0},
#         ),
#     ]
#     for pred, actual, expected_score in test_case:
#         assert bleu.compute(pred, actual) == expected_score


def test_sacrebleu(sacrebleu):
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
    for pred, actual, _ in test_case:
        assert sacrebleu.compute(pred, actual) == 0.0


def test_syllables(lexical):
    assert lexical.syllables("word") > 0


def test_lexical_lex_match_1(lexical, pos_sequence_1):
    scores = lexical.lex_match_1(pos_sequence_1)
    assert len(scores) > 0


def test_lexical_lex_match_2(lexical, pos_sequence_2):
    scores = lexical.lex_match_2(pos_sequence_2)
    assert len(scores) > 0


def test_lexical_basic_stats(lexical):
    article = "The quick brown fox jumped over the lazy dog. He moved and jumped."
    lexical.basic_stats(article)
