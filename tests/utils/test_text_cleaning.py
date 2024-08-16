from specialk.core.sanitisation import fix_unicode, normalize_punctuation
import pytest
from specialk.datasets.filters import PreTrainingFilter, is_valid_text, CLEAN, REASON
import pandas as pd
from specialk.core.constants import URL_TOKEN


def test_fix_unicode():
    test_cases = [
        (
            "Etant donnÃ© que le poisson est excellent",
            "Etant donné que le poisson est excellent",
        ),
        ("Ritchie\u0027s house is nearby", "Ritchie's house is nearby"),
        ("Caf&eacute; au lait", "Café au lait"),
        ("Caf%C3%A9 au lait", "Café au lait"),
        ("S&amp;P 500", "S&P 500"),
        ("$10,000 Gold?", "$10,000 Gold?"),
        ("I'm that guy pal", "I'm that guy pal"),
        # New test cases
        (
            "&#x2022; Vous travaillez sur la partie statistique",
            "• Vous travaillez sur la partie statistique",
        ),
        (
            "http://www.facebook.com/album.php?aid=2511446&id;=1209158&l;=4e976d2540",
            URL_TOKEN,
        ),
        (
            "check out my website at http://www.facebook.com/album.php?aid=2511446&id;=1209158&l;=4e976d2540 you're going to see a lot of cool things",
            f"check out my website at {URL_TOKEN} you're going to see a lot of cool things",
        ),
        (
            "Our guests can profit from the services of the Hotel Salina Maris, which is 300m away from the B&B IM GRÜNEN. You will find there the reception, the breakfast room (breakfast +CHF 11.00), the thermal brine pool (33&degr; C; free entrance)",
            "Our guests can profit from the services of the Hotel Salina Maris, which is 300m away from the B&B IM GRÜNEN. You will find there the reception, the breakfast room (breakfast +CHF 11.00), the thermal brine pool (33° C; free entrance)",
        ),
        (
            "&lquot;Mistral&lquot; wind cannot but charm you.",
            '"Mistral" wind cannot but charm you.',
        ),
        ('&uot;Shopaholics" and fashion lovers', '"Shopaholics" and fashion lovers'),
        (
            "Many of these composers are unknown outside Belgium &emdash; one of the specific purposes",
            "Many of these composers are unknown outside Belgium - one of the specific purposes",
        ),
        (
            "Quelques aspects de la dynamique des systêmes plan&eagrave;taires extrasolaires.",
            "Quelques aspects de la dynamique des systêmes planètaires extrasolaires.",
        ),
        (
            "defining the list of words cl&eacute;s page optimization",
            "defining the list of words clés page optimization",
        ),
    ]

    for src, tgt in test_cases:
        fix = fix_unicode(src)
        assert fix == tgt, f"Expected '{tgt}', but got '{fix}' for input '{src}'"


def test_normalize_punctuation():
    test_cases = [
        (" “ s’européanisent ” à ", ' " s\'européanisent " à '),
        ("$10,000 Gold?", "$10,000 Gold?"),
        ("I'm that guy pal", "I'm that guy pal"),
        ("Caf%C3%A9 au lait", "Caf%C3%A9 au lait"),
        ("S&amp;P 500", "S&amp;P 500"),
    ]
    for src, tgt in test_cases:
        fix = normalize_punctuation(src)
        assert fix == tgt


# Test cases for is_valid_text function
@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello, world!", True),
        ("", False),
        (" ", False),
        ("12345678901234567890", False),
        ("Too many periods............", False),
        ("Too many parentheses ((((((((((", False),
        ("Too many commas,,,,,,,,,", False),
        ("Too many @ symbols @@@@@@@@@@", False),
        ("Normal text with some numbers 123", True),
        ("This is a valid sentence.", True),
    ],
)
def test_is_valid_text(text, expected):
    assert is_valid_text(text) == expected


# Test cases for PreTrainingFilter class
def test_pretraining_filter_sanity_check():
    # Create a sample DataFrame
    df = pd.DataFrame(
        {
            "source": [
                "Hello, world!",
                "Invalid @@@@@@@@@@",
                "Valid text",
                "12345678901234567890",
            ],
            "target": [
                "Bonjour, monde!",
                "Valid text",
                "Invalid ,,,,,,,,,,",
                "98765432109876543210",
            ],
            CLEAN: [True] * 4,
            REASON: [""] * 4,
        }
    )

    # Initialize PreTrainingFilter
    ptf = PreTrainingFilter(df, "source", "target")

    # Run sanity check
    ptf.filter_sanity_check()

    # Check results
    expected_clean = [True, False, False, False]
    expected_reasons = [
        None,
        "src fails sanity check",
        "tgt fails sanity check",
        "src fails sanity check",
    ]

    pd.testing.assert_series_equal(ptf.df[CLEAN], pd.Series(expected_clean, name=CLEAN))
    pd.testing.assert_series_equal(
        ptf.df[REASON], pd.Series(expected_reasons, name=REASON)
    )


def test_pretraining_filter_empty_dataframe():
    df = pd.DataFrame(columns=["source", "target", CLEAN, REASON])
    ptf = PreTrainingFilter(df, "source", "target")
    ptf.filter_sanity_check()
    assert len(ptf.df) == 0


def test_pretraining_filter_all_valid():
    df = pd.DataFrame(
        {
            "source": ["Valid text 1", "Valid text 2", "Valid text 3"],
            "target": ["Valid text A", "Valid text B", "Valid text C"],
            CLEAN: [True] * 3,
            REASON: [None] * 3,
        }
    )
    ptf = PreTrainingFilter(df, "source", "target")
    ptf.filter_sanity_check()
    assert ptf.df[CLEAN].all()
