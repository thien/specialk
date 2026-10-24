from specialk.models.tokenizers.moses import MosesPreTokenizer


def test_moses_pretokeniser_matches_2019_settings():
    tokenize = MosesPreTokenizer("en")
    assert tokenize("Hello, world!") == "hello , world !"
    assert tokenize("well-known fact") == "well @-@ known fact"
    assert tokenize("This is a test.") == "this is a test ."


def test_moses_pretokeniser_french():
    assert MosesPreTokenizer("fr")("l'État français") == "l' état français"
