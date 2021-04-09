from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

if __name__ == '__main__':
    count_vect = CountVectorizer()
    x_train_counts = count_vect.fit()
    print("Ay que fabuloso")