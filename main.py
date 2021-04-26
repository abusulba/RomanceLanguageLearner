from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import nltk
import os
import random

class RomanceLanguageClassifier:
    def __init__(self):
        self.french_lines = self.lines_extractor('textData/fra_news_2010_10K-sentences.txt').lower().splitlines()
        self.italian_lines = self.lines_extractor('textData/ita_news_2010_10K-sentences.txt').lower().splitlines()
        self.portuguese_lines = self.lines_extractor('textData/por_news_2020_10K-sentences.txt').lower().splitlines()
        self.spanish_lines = self.lines_extractor('textData/spa_news_2010_10K-sentences.txt').lower().splitlines()
        self.train(self.spanish_lines, self.french_lines, self.italian_lines, self.portuguese_lines)
        
        # language tokens are the unique set of tokens from each language
        self.french_tokens = self.language_tokens(self.french_lines)
        self.italian_tokens = self.language_tokens(self.italian_lines)
        self.portuguese_tokens = self.language_tokens(self.portuguese_lines)
        self.spanish_tokens = self.language_tokens(self.spanish_lines)
        exact_sets = []

        # set of tokens shared by all languages (mostly proper nouns and other names)
        # we get the set difference of the intersections with all_shared to get more unique, language-specific matches
        self.all_shared = self.italian_tokens & self.portuguese_tokens & self.spanish_tokens & self.french_tokens

        #define exact cognates (french-other)
        self.fre_esp_exact = list((self.french_tokens & self.spanish_tokens)- (self.all_shared))
        print(len(self.fre_esp_exact))
        self.fre_ita_exact = list((self.french_tokens & self.italian_tokens)- (self.all_shared))
        print(len(self.fre_ita_exact))
        self.fre_por_exact = list((self.french_tokens & self.portuguese_tokens)- (self.all_shared))
        print(len(self.fre_por_exact))

        #define exact cognates (spanish-other)
        # spanish-french == self.fre_esp_exact (intersection is equal)
        self.esp_ita_exact = list((self.spanish_tokens & self.italian_tokens) - (self.all_shared))
        print(len(self.esp_ita_exact))
        self.esp_por_exact = list(self.spanish_tokens & self.portuguese_tokens - (self.all_shared))
        print(len(self.esp_por_exact))      # --- ESP and POR have MAX similarity

        #define exact cognates (italian-other)
        # self.ita_esp_exact = self.esp_ita_exact (intersection is equal)
        # self.ita_fre_exact = self.fre_ita_exact (intersection is equal)
        self.ita_por_exact = list(self.italian_tokens & self.portuguese_tokens) - (self.all_shared))
        print(len(self.ita_por_exact))
        

    def lines_extractor(self, file_name):
        output = ''
        if os.path.exists(file_name):  # file already exists
            with open(file_name, 'r', encoding='utf-8') as file:
                output = file.read()
        return output

    def line_features(self, line):
        tokens = nltk.word_tokenize(line)
        featureset = {}
        for word in tokens:
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

    def train(self, spanish, french, italian, portuguese):
        dataset = []
        for line in spanish:
            dataset.append((self.line_features(line), 'spanish'))
        for line in french:
            dataset.append((self.line_features(line), 'french'))
        for line in italian:
            dataset.append((self.line_features(line), 'italian'))
        for line in portuguese:
            dataset.append((self.line_features(line), 'portuguese'))

        size = int(.9 * len(dataset))
        random.shuffle(dataset)
        test = dataset[size:]
        train = dataset[:size]

        self.classifier = nltk.NaiveBayesClassifier.train(train)
        print(nltk.classify.accuracy(self.classifier, test))
        # print(self.classifier.show_most_informative_features(100))
        # takes in 4 lists of lines of each language
        # calls line features on each line of the 4 languages
        # stores a list of tuples
        #   --> (line_features(line), language)
        #
        # combine list of tuples for each language and shuffle
        # initialize classifier and train on 70%? of the data

    def predict(self, prediction_text):
        x = 6
        return self.classifier.classify(self.line_features(prediction_text))
        # uses classifier to predict which language the given text is in
        #
        
    def language_tokens(self, language_lines):
        # tokens of the language
        token_list = []
        for line in language_lines:
            tokens = nltk.word_tokenize(line)
            # remove numeric tokens
            tokens = [token for token in tokens if token.isnumeric() == False]
            token_list += tokens
        #print(len(set(token_list)))
        return set(token_list)

    def lexical_distance(self, s1, s2):
        if len(s1) < len(s2):
            return lexical_distance(s2, s1)   # longer string goes first

        # len(s1) >= len(s2)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)  # Dynamic Programming based
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
                deletions = current_row[j] + 1       # than s2
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

if __name__ == '__main__':
    count_vect = CountVectorizer()
    # x_train_counts = count_vect.fit()
    rl = RomanceLanguageClassifier()
    in_ = ''
    while in_ != 'q':
        in_ = input('Please enter text to predict: ')
        print(rl.predict(in_))
    
    print(rl.predict('yo soy una mujer muy inteligente'))


