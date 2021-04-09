from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import os

if __name__ == '__main__':
    count_vect = CountVectorizer()
    x_train_counts = count_vect.fit()
    print("Ay que fabuloso")

class RomanceLanguageClassifier:
    def __init__(self):

    def lines_extractor(self, file_name):
        output = ''
        if os.path.exists(file_name):  # file already exists
            with open(file_name, 'r') as file:
                output = file.read()
        return output

    def line_features(self, line):
        # line features include:
        #   contains{word} = true
        #   existence of language specific characters
        #       ç = French/Portuguese
        #       â = French

    def train(self, spanish, french, italian, portugues):
        # takes in 4 lists of lines of each language
        # calls line features on each line of the 4 languages
        # stores a list of tuples
        #   --> (line_features(line), language)
        #
        # combine list of tuples for each language and shuffle
        # initialize classifier and train on 70%? of the data

    def predict(self, prediction_text):
        # uses classifier to predict which language the given text is in
        #

if __name__ == '__main__':
    print("Ay que fabuloso")
