import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import gutenberg

def main():
    path = './corpus'
    titles, corpus = [], []
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        titles.append(filename)
        with open(file_path, 'r') as f:
            corpus.append(f.read().replace('\n', ''))
    titles = gutenberg.fileids()
    corpus = [gutenberg.raw(title) for title in titles]
    vectorizer = TfidfVectorizer(max_df=.9, stop_words='english')
    weights = vectorizer.fit_transform(corpus).toarray()

    similarity = cosine_similarity(weights)
    most_similar = np.argmax(similarity, axis=0)
    for index, title in enumerate(titles):
        print("{}: {}".format(title, (list(similarity[index]))))

if __name__ == '__main__':
    main()
