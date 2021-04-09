from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import nltk
import os

class RomanceLanguageClassifier:
    def __init__(self):
        self.french_lines = self.lines_extractor('textData/fra_news_2010_10K-sentences.txt').splitlines()
        self.italian_lines = self.lines_extractor('textData/ita_news_2010_10K-sentences.txt').splitlines()
        self.portuguese_lines = self.lines_extractor('textData/por_news_2010_10K-sentences.txt').splitlines()
        self.spanish_lines = self.lines_extractor('textData/spa_news_2010_10K-sentences.txt').splitlines()

    def lines_extractor(self, file_name):
        output = ''
        if os.path.exists(file_name):  # file already exists
            with open(file_name, 'r', encoding='utf-8') as file:
                output = file.read()
        return output

    def line_features(self, line):
        tokens = nltk.word_tokenize(line)
        featureset = {}
        for word in line:
            featureset['contains({0})'.format(word)] = True
        featureset['çedille'] = ('ç' in line)
        featureset['grave'] = any(accent in line for accent in ('à', 'è', 'ù'))
        featureset['circumflex'] = any(accent in line for accent in ('â', 'ê', 'î', 'ô', 'û'))
        featureset['acute'] = any(accent in line for accent in ('á', 'é', 'í', 'ó', 'ú'))
        featureset['spanish-special'] = any(accent in line for accent in ('ñ', '¿', '¡'))
        featureset['tildes'] = any(accent in line for accent in ('ã', 'õ'))
        return featureset

        # line features include:
        #   contains{word} = true
        #   existence of language specific characters
        #       ç = French/Portuguese
        #       â = French

    def train(self, spanish, french, italian, portugues):
        x = 5
        # takes in 4 lists of lines of each language
        # calls line features on each line of the 4 languages
        # stores a list of tuples
        #   --> (line_features(line), language)
        #
        # combine list of tuples for each language and shuffle
        # initialize classifier and train on 70%? of the data

    def predict(self, prediction_text):
        x = 6
        # uses classifier to predict which language the given text is in
        #

if __name__ == '__main__':
    count_vect = CountVectorizer()
    # x_train_counts = count_vect.fit()
    rl = RomanceLanguageClassifier()
    print(len(rl.spanish_raw))

