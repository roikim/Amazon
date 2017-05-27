import nltk
from nltk.corpus import treebank
import nltk.collocations
import nltk.corpus
import collections




class Tokens():

    def __init__(self, sentence):
        self.sentence = sentence
        self.tokens = []
        self.tagged = []
        self.tree =[]

    # def __init__(self):
    #     self.sentence = ""
    #     self.tokens = []

    def tokenizer(self):
        self.tokens = nltk.word_tokenize(self.sentence)
        #print self.tokens
        self.tagged = nltk.pos_tag(self.tokens)
        #print self.tagged
        self.entities = nltk.chunk.ne_chunk(self.tagged)
        #print self.entities


    def create_ngram(self):
        bgm = nltk.collocations.BigramAssocMeasures()
        finder = nltk.collocations.BigramCollocationFinder.from_words(
            nltk.corpus.brown.words())
        scored = finder.score_ngrams(bgm.likelihood_ratio)

        # Group bigrams by first word in bigram.
        prefix_keys = collections.defaultdict(list)
        for key, scores in scored:
            prefix_keys[key[0]].append((key[1], scores))

        # Sort keyed bigrams by strongest association.
        for key in prefix_keys:
            prefix_keys[key].sort(key=lambda x: -x[1])

        #print self.tokens[2], prefix_keys[self.tokens[2]][:5]

    def draw_tree(self):
        self.tree = treebank.parsed_sents('wsj_0001.mrg')[0]
        # self.tree.draw()



def main():
    tokens = Tokens("at eight and 20 I go to the market and I didn't")
    tokens.tokenizer()
    tokens.draw_tree()
    tokens.create_ngram()


if __name__ == "__main__":
    main()










