"""
DOCSTRING
"""
import os
import sklearn
# third-party
import pandas as pd
import nltk
# first-party
import kaggle_word_vec_utility


train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'),
                    header=0, delimiter="\t", quoting=3)

test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'),
                   header=0, delimiter="\t", quoting=3)

print('The first review is:')
print(train["review"][0])
input("Press Enter to continue . . .")

print('Download text data sets.')
nltk.download()
clean_train_reviews = []

print("Cleaning and parsing the training set movie reviews.")
for i in range(len(train["review"])):
    clean_train_reviews.append(" ".join(
        kaggle_word_vec_utility.KaggleWord2VecUtility.review_to_wordlist(train["review"][i],
                                                                         True)))

print("Creating the bag of words.")
vectorizer = sklearn.feature_extraction.text.CountVectorizer(analyzer="word",
                                                             tokenizer=None,
                                                             preprocessor=None,
                                                             stop_words=None,
                                                             max_features=5000)

train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.to_array()

print("Training the random forest.")
forest = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data_features, train["sentiment"])
clean_test_reviews = []

print("Cleaning and parsing the test set movie reviews.")
for i in range(0, len(test["review"])):
    clean_test_reviews.append(" ".join(
        kaggle_word_vec_utility.KaggleWord2VecUtility.review_to_wordlist(test["review"][i],
                                                                         True)))
test_data_features = vectorizer.transform(clean_test_reviews)

print("Predicting test labels.")
result = forest.predict(test_data_features)
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model.csv'),
              index=False,
              quoting=3)
print("Wrote results to Bag_of_Words_model.csv")
