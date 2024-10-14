from __future__ import annotations

import specialk.metrics.style_transfer.style_lexicon as stme_lexicon
import specialk.metrics.style_transfer.utils as stme_utils
from specialk.core.utils import log


class StyleLexicon:
    def __init__(self):
        self.lexicon: dict[str, tuple] = {}

    def create(self, src: list[str], tgt: list[str]) -> dict[str, tuple]:
        """
        finds style lexicon.
        Returns a set of words following a particular style.
        """
        styles = {0: "styles"}
        # create vectoriser and inverse vocab.
        x, y = stme_utils.compile_binary_dataset(src, tgt)
        vectoriser = stme_lexicon.fit_vectorizer(x)
        inv_vocab = stme_utils.invert_dict(vectoriser.vocabulary_)

        # train style weights model
        src_weights = vectoriser.transform(x)
        model = stme_lexicon.train("l1", 3, src_weights, y)

        # extract style features and weights
        nz_weights, f_nums = stme_lexicon.extract_nonzero_weights(model)
        self.lexicon = stme_lexicon.collect_style_features_and_weights(
            nz_weights, styles, inv_vocab, f_nums
        )
        log.info("Created Style Lexicon.")

    @classmethod
    def from_json(cls, filepath) -> StyleLexicon:
        lex = cls()
        lex.lexicon = stme_utils.load_json(filepath)
        return lex

    def save(self, filepath):
        stme_utils.save_json(self.lexicon, filepath)
