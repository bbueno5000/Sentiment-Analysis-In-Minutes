"""
DOCSTRING
"""
# standard
import re
# non-standard
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

class KaggleWord2VecUtility:
    """
    KaggleWord2VecUtility is a utility class for processing
    raw HTML text into segments for further learning.
    """
    @staticmethod
    def review_to_wordlist(review, remove_stopwords=False):
        """
        Function to convert a document to a sequence of words,
        optionally removing stop words.  Returns a list of words.
        """
        review_text = BeautifulSoup(review).get_text()
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        words = review_text.lower().split()
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        return(words)

    @staticmethod
    def review_to_sentences(review, tokenizer, remove_stopwords=False):
        """
        Function to split a review into parsed sentences. 
        Returns a list of sentences, where each sentence is a list of words
        """
        raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sentences.append(KaggleWord2VecUtility.review_to_wordlist(raw_sentence,
                                                                          remove_stopwords))
        return sentences
