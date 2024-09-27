import unicodedata
import re
from urllib.parse import unquote
import html
from specialk.core import log
import ftfy
from specialk.core.constants import URL_TOKEN

URL_PATTERN = re.compile(
    "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)
HTML_ENTITY_PATTERN = re.compile("&([a-z#0-9]+);")

# Punctuation normalization mappings
PUNCT_MAPPINGS = {
    "“": '"',  # Left double quotation mark
    "”": '"',  # Right double quotation mark
    "‘": "'",  # Left single quotation mark
    "’": "'",  # Right single quotation mark
    "…": "...",  # Ellipsis
    "–": "-",  # En dash
    "—": "-",  # Em dash
    "„": '"',  # Double low-9 quotation mark
    "«": '"',  # Left-pointing double angle quotation mark
    "»": '"',  # Right-pointing double angle quotation mark
    "′": "'",  # Prime
    "″": '"',  # Double prime
    "≥": ">=",  # Greater-than or equal to
    # "•": "*",  # Bullet
    # "°": " degrees",  # Degree sign
}

ONE_OFF_MAPPINGS = {
    "&degr;": "°",
    "&lquot;": '"',
    "&rquot;": '"',
    "&quot;": '"',
    "&uot;": '"',
    "&emdash;": "-",
    "&eagrave;": "&egrave;",
    "&eacutes;": "&eacute;",
}

# Create a regex pattern for punctuation
PUNCT_PATTERN = re.compile("|".join(map(re.escape, PUNCT_MAPPINGS.keys())))
HAMMER_PATTERN = re.compile("|".join(map(re.escape, ONE_OFF_MAPPINGS.keys())))

sequence_limits = {
    ".": 8,
    "(": 8,
    ")": 8,
    ",": 8,
    "@": 8,
}


def is_valid_text(x: str) -> bool:
    """
    Heuristic based text validity, gopher style.

    (yeah, heuristic based filtering has been
    around for ages, I know).
    """
    if len(x.strip()) < 1:
        return False

    # if the text contains a lot of numbers,
    if sum(c.isdigit() for c in x) / len(x) > 0.4:
        return False

    # check ratio of characters.
    base = {}
    for char in x:
        if char not in base:
            base[char] = 0
        base[char] += 1
        if char in sequence_limits:
            if sequence_limits[char] < base[char]:
                return False
    return True


def normalize_punctuation(text: str) -> str:
    """Normalize Unicode punctuation marks to ASCII equivalents. This
    should only be applied on english."""
    return PUNCT_PATTERN.sub(lambda m: PUNCT_MAPPINGS[m.group(0)], text)


def hammer_replacement(text: str) -> str:
    return HAMMER_PATTERN.sub(lambda m: ONE_OFF_MAPPINGS[m.group(0)], text)


def can_parse_html_pattern(text: str) -> bool:
    try:
        HTML_ENTITY_PATTERN.sub(
            lambda m: chr(
                int(m.group(1)[1:])
                if m.group(1).startswith("#")
                else ord(html.entities.html5[m.group(1)])
            ),
            text,
        )
        return True
    except Exception:
        return False


def fix_unicode(text: str) -> str:
    text = URL_PATTERN.sub(URL_TOKEN, text)
    # Use ftfy to fix and normalize Unicode
    text = ftfy.fix_text(text)

    text = hammer_replacement(text)
    # Handle HTML entities.
    try:
        text = HTML_ENTITY_PATTERN.sub(
            lambda m: chr(
                int(m.group(1)[1:])
                if m.group(1).startswith("#")
                else ord(html.entities.html5[m.group(1)])
            ),
            text,
        )
    except KeyError as e:
        log.error(
            "Could not parse HTML_ENTITY_PATTERN for text, so skipping",
            text=text,
            type="KeyError",
            error=str(e),
        )
    except ValueError as e:
        log.error(
            "Could not parse HTML_ENTITY_PATTERN for text, so skipping",
            text=text,
            type="ValueError",
            error=str(e),
        )

    # Handle percent-encoded characters.
    text = unquote(text)

    # Attempt Latin-1 to UTF-8 conversion.
    try:
        text = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    # Handle Unicode escape sequences.
    try:
        text = text.encode("ascii").decode("unicode-escape")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    # Normalize Unicode (combine characters and their diacritics).
    text = unicodedata.normalize("NFC", text)

    # Remove any invalid Unicode characters.
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")

    return text
