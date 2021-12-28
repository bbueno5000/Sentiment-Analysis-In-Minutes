"""
DOCSTRING
"""
import bs4
import nltk
import os
import pandas
import re
import sklearn

class KaggleWord2VecUtility:
    """
    This class is a utility class for processing
    raw HTML text into segments for further learning.
    """
    @staticmethod
    def review_to_sentences(review, tokenizer, remove_stopwords=False):
        """
        This function splits a review into parsed sentences. 
        Returns a list of sentences, where each sentence is a list of words
        """
        raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
        sentences = list()
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sentences.append(
                    KaggleWord2VecUtility.review_to_wordlist(
                        raw_sentence, remove_stopwords))
        return sentences
    
    @staticmethod
    def review_to_wordlist(review, remove_stopwords=False):
        """
        This function converts a document to a sequence of words,
        optionally removing stop words.
        Returns a list of words.
        """
        review_text = bs4.BeautifulSoup(review).get_text()
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        words = review_text.lower().split()
        if remove_stopwords:
            stops = set(nltk.corpus.stopwords.words("english"))
            words = [w for w in words if not w in stops]
        return(words)

class SentimentAnalysis:
    """
    DOCSTRING
    """
    def __init__(self):
        """
        DOCSTRING
        """
        train = pandas.read_csv('data\\ulabeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
        test = pandas.read_csv('data\\testData.tsv', header=0, delimiter="\t", quoting=3)
        nltk.download()
        self.clean_train_reviews = list()
        for i in range(len(train['review'])):
            self.clean_train_reviews.append(
                ' '.join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))
        
    def __call__(self):
        """
        DOCSTRING
        """
        vectorizer = sklearn.feature_extraction.text.CountVectorizer(
            analyzer='word', tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
        train_data_features = vectorizer.fit_transform(self.clean_train_reviews)
        train_data_features = train_data_features.to_array()
        forest = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
        forest = forest.fit(train_data_features, train['sentiment'])
        clean_test_reviews = list()
        for i in range(len(test['review'])):
            clean_test_reviews.append(
                ' '.join(KaggleWord2VecUtility.review_to_wordlist(test['review'][i], True)))
        test_data_features = vectorizer.transform(clean_test_reviews)
        result = forest.predict(test_data_features)
        output = pandas.DataFrame(data={'id':test['id'], 'sentiment':result})
        output.to_csv('data\\Bag_of_Words_model.csv', index=False, quoting=3)

if __name__ == '__main__':
    SentimentAnalysis()()
