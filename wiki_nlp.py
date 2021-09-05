"""
A way to test your knowledge, make a script that can take an input from the command line, search for a Wikipedia
article, turn the article into a document and return all token's lemmas and parts of speech in python. That was
literally my first ever NLP project, except it had to also return all hypernyms (which is not a built-in feature and I
had to build myself). You can probably google a Stack Overflow answer to this, but I encourage you not to, and to
actually get through all three of the above resources and attempt from scratch. Completing the spaCy course in #3 will
give you all of the tools you need to be able to complete this test.

>>> ./wikipedia_nlp.py <arbitrary_word>
# Read in arbitrary word from CLI, search Wikipedia (I assume there is an API, python module, or something along those lines—I can do some Googling to find out how)
# Load the page content from Wikipedia
# Use Spacy to turn the wiki corpus/text into a Doc object and print out tokens’ lemmas and POS in some fashion
# Bonus points to also include hypernyms

"""

import argparse
import spacy
import os
import pandas
import wikipedia


SPACY_LANG_MODEL = 'en_core_web_sm'


class WikipediaNLP:
    def __init__(self, search_word):
        self.word = search_word
        self.corpus = None
        self.nlp = None
        self.nlp_doc = None

    def run(self, show_all_results=False, save=False):
        self.corpus = self.get_corpus_from_wikipedia_page()
        # Download the language model if not present, then load it in
        spacy.cli.download(SPACY_LANG_MODEL)
        self.nlp = spacy.load(SPACY_LANG_MODEL)
        self.nlp_doc = self.process_corpus()
        self.print_tokens(show_all_results=show_all_results, save=save)

    def get_corpus_from_wikipedia_page(self):
        pages = wikipedia.search(self.word)
        if len(pages) == 0:
            raise Exception(f'No pages found for "{self.word}"')
        if len(pages) > 1:
            print(f'Found multiple options for "{self.word}", using "{pages[0]}." Other options: {pages[1:]}')
        wiki_page = wikipedia.page(pages[0])
        print(f'Pulling information from {wiki_page.url}')
        return wiki_page.content

    def process_corpus(self):
        print('Processing corpus')
        return self.nlp(self.corpus)

    def print_tokens(self, save=False, file_name=None, show_all_results=False):
        if file_name is None:
            file_name = f'wiki_nlp_{self.word}.csv'
        chart = pandas.DataFrame(
            {'Token': self.nlp_doc,
             'Lemma': [t.lemma_ for t in self.nlp_doc],
             'POS': [t.pos_ for t in self.nlp_doc]
             }
        )
        max_rows = pandas.get_option('display.max_rows')
        if show_all_results:
            pandas.set_option('display.max_rows', chart.shape[0] + 1)
        print(chart)
        pandas.set_option('display.max_rows', max_rows)

        if save:
            filepath = os.path.abspath(os.path.join('.', file_name))
            chart.to_csv(filepath)
            print(f'Saved dataframe to {filepath}')


def get_cli_args():
    arg_parser = argparse.ArgumentParser('Process a wikipedia page')
    arg_parser.add_argument('word')
    arg_parser.add_argument('--all', '-a', action='store_true')
    arg_parser.add_argument('--save', '-s', action='store_true')

    return arg_parser.parse_args()


if __name__ == '__main__':
    # Read input from the command line
    args = get_cli_args()

    print(f'Arbitrary word: {args.word}')
    wiki_nlp = WikipediaNLP(args.word)
    wiki_nlp.run(args.all, args.save)
