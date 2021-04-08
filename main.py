def lines_extractor(file_name):
    # Returns list of lines from .txt file

def line_features(line):
    # line features include:
    #   contains{word} = true
    #   existence of language specific characters
    #       ç = French/Portuguese
    #       â = French

def train(spanish, french, italian, portugues):
    # takes in 4 lists of lines of each language
    # calls line features on each line of the 4 languages
    # stores a list of tuples
    #   --> (line_features(line), language)
    #
    # combine list of tuples for each language and shuffle
    # initialize classifier and train on 70%? of the data

def predict(prediction_text):
    # uses classifier to predict which language the given text is in
    #

if __name__ == '__main__':
    print("Ay que fabuloso")
