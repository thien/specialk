import spacy
import torch

class Metrics:
    def __init__(self):
        self.running = True

    def load_glove(self):
        """
        Loads glove dataset.
        """
        return None

    def load_spacy(self):
        # load spacy model if it isn't in memory yet.
        if not self.spacy:
            self.spacy = spacy.load('en_core_web_sm')

    def lex_match_1(tokens):
        """
        finds ``it v-link ADJ finite/non-finite clause''
        
        eg:
            "It's unclear what Teresa May is planning."
        
        params:
            tokens: tokenized sentence from nlp(sentence)
        returns:
            matches: None if nothing is found,
                    [(match pairs)] otherwise.
        """
        matches = []

        index_limit = len(tokens)
        index = 0
        while index < index_limit:
            token = tokens[index]
            if token.text.lower() == "it":
                if tokens[index+1].pos_ == "VERB" and tokens[index+2].pos_ == "ADJ":
                    matches.append((index, (token, tokens[index+1], tokens[index+2])))
                    index = index + 2
            index += 1
            
        return matches if matches else None

    def lex_match_2(tokens):
        """
        finds ``v-link ADJ prep''
        
        eg:
            "..he was responsible for all for.."
        
        params:
            tokens: tokenized sentence from nlp(sentence)
        returns:
            matches: None if nothing is found,
                    [(match pairs)] otherwise.
        """
        matches = []

        index_limit = len(tokens)
        index = 0
        while index < index_limit:
            token = tokens[index]
            if token.pos_ == "VERB":
                group = [token]
                next_index = index+1
                # detect any adverbs before adj and adp.
                # e.g. "be *very, very,* embarrassing.."
                while tokens[next_index].pos_ == "ADV":
                    group.append(tokens[next_index])
                    next_index += 1
        
                if tokens[next_index].pos_ == "ADJ" and tokens[next_index+1].pos_ == "ADP":
                    group.append(tokens[next_index])
                    group.append(tokens[next_index+1])
                    matches.append((index, tuple(group)))
                    index = next_index + 2
            index += 1
            
        return matches if matches else None

    def earth_movers_distance(self):
        """
        compute earth mover's distance between files.
        """
        return None

    def word_movers_distance(self):
        """
        compute word mover's distance between files.
        """
        return None

    def bleu(self):
        return None
    
    def rouge(self):
        return None

    def meteor(self):
        return None

    def readability_score(self):
        return None

    # style transfer intensity
    # content preservation
    # naturalness



if __name__ == "__main__":
