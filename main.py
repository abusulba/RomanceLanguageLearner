from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from googletrans import Translator
from PyDictionary import PyDictionary
from collections import Counter

import nltk
import os
import random
import math


class RomanceLanguageClassifier:
    def __init__(self):
        self.french_lines = self.lines_extractor('textData/fra_news_2010_10K-sentences.txt').lower().splitlines()
        self.italian_lines = self.lines_extractor('textData/ita_news_2010_10K-sentences.txt').lower().splitlines()
        self.portuguese_lines = self.lines_extractor('textData/por_news_2020_10K-sentences.txt').lower().splitlines()
        self.spanish_lines = self.lines_extractor('textData/spa_news_2010_10K-sentences.txt').lower().splitlines()
        self.train(self.spanish_lines, self.french_lines, self.italian_lines, self.portuguese_lines)

        self.translator = Translator()  # create instance of googletrans module
        self.dictionary = PyDictionary()

        self.noun_cognates = self.loadNounCognates()
        
    def token_info(self):
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
        self.ita_por_exact = list((self.italian_tokens & self.portuguese_tokens) - (self.all_shared))
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
        return self.classifier.classify(self.line_features(prediction_text))
        # uses classifier to predict which language the given text is in
        
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
            return self.lexical_distance(s2, s1)   # longer string goes first

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
    
    def cognate_score(self, word1, word2, known_cognate=False):
        final_score = 0
        meaning_score = 0
        spelling_score = 0

        en_word1 = self.translator.translate(word1).text
        en_word2 = self.translator.translate(word2).text
        
        # print(en_word1 + " " + en_word2)
        if not known_cognate and en_word1 != en_word2:   # if meaning is not exact same, return 0
            return 0

        dist = self.lexical_distance(word1, word2)   # finds how lexically similar words are
        
        longer = ''
        if len(word1) > len(word2):
            longer = word1
        else:
            longer = word2
        
        spelling_score = 1 - (dist / len(longer))      # final score based on ratio of longer word

        # print("spelling score: " + str(spelling_score))
        
        return spelling_score
    
    def is_cognate(self, word1, word2, known_cognate=False):
        score = self.cognate_score(word1, word2, known_cognate)
        threshold = .4
        if score > threshold:
            return True
        
        return False

    def loadNounCognates(self):
        file = open(r"nounCognates.txt", "r", encoding='utf-8')
        lines = file.readlines()
        updated = []
        final = []
        #remove tab and newline characters
        for line in lines:
            temp = line.replace('\t', ' ')
            temp = temp.replace('\n', '')
            updated.append(temp)
        #correct updated
        for line in updated:
            tokens = line.split(' ')
            tokens = [token for token in tokens if len(token) > 2 and token[0] != '(' and token[-1] != ')']
            final.append(tokens)
        #final format = list of lists of cognates
        #Language Order = ['ENGLISH', 'FRENCH', 'ITALIAN', 'SPANISH', 'PORTUGUESE']
        return final
    
    def cognate_information(self, cognates):
        length = len(cognates)
        fr_it_score = 0
        fr_es_score = 0
        fr_po_score = 0
        it_es_score = 0
        it_po_score = 0
        es_po_score = 0

        for cognate in cognates:
            if len(cognate) < 5:
                continue
            
            en_word = cognate[0]
            fr_word = cognate[1]
            it_word = cognate[2]
            es_word = cognate[3]
            po_word = cognate[4]

            fr_it_score += self.score_val(self.is_cognate(fr_word, it_word, known_cognate=True))
            fr_es_score += self.score_val(self.is_cognate(fr_word, es_word, known_cognate=True))
            fr_po_score += self.score_val(self.is_cognate(fr_word, po_word, known_cognate=True))
            it_es_score += self.score_val(self.is_cognate(it_word, es_word, known_cognate=True))
            it_po_score += self.score_val(self.is_cognate(it_word, po_word, known_cognate=True))
            es_po_score += self.score_val(self.is_cognate(es_word, po_word, known_cognate=True))
    
        fr_it_score /= length
        fr_es_score /= length
        fr_po_score /= length
        it_es_score /= length
        it_po_score /= length
        es_po_score /= length

        print("French and Italian overlap: {0}".format(fr_it_score))
        print("French and Spanish overlap: {0}".format(fr_es_score))
        print("French and Portuguese overlap: {0}".format(fr_po_score))
        print("Italian and Spanish overlap: {0}".format(it_es_score))
        print("Italian and Portuguese overlap: {0}".format(it_po_score))
        print("Spanish and Portuguese overlap: {0}".format(es_po_score))


    def score_val(self, check):   # helper function to simplify code for cognate data
        if check:
            return 1
        else:
            return 0

        


if __name__ == '__main__':
    # count_vect = CountVectorizer()
    # x_train_counts = count_vect.fit()
    rl = RomanceLanguageClassifier()
    in_ = ''
    while in_ != 'q':
        in_ = input('Please enter text to predict: ')
        print(rl.predict(in_))
    rl.cognate_information(rl.noun_cognates)
    translator = Translator()
    # print(rl.cognate_score('chico', 'bambino'))
    # print(rl.predict('yo soy una mujer muy inteligente'))


