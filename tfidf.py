import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

def main():
    pattern = './corpus/*.txt'
    files = glob.glob(pattern)
    vectorizer = TfidfVectorizer(input='filename', max_df=.9, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(files)

    similarity = cosine_similarity(tfidf_matrix)
    for index, file in enumerate(files):
        print("{}: {}".format(file, (list(similarity[index]))))

    dense_tfidf_matrix = tfidf_matrix.toarray()
    model = TSNE(n_components=2, random_state=0, metric='cosine', learning_rate=150)
    points = model.fit_transform(dense_tfidf_matrix)

    df = pd.DataFrame(points, index=files, columns=['x', 'y'])
    fig, ax = plt.subplots()
    df.plot('x', 'y', kind='scatter', ax=ax)
    for k, v in df.iterrows():
        ax.annotate(k, v)
    plt.show()

if __name__ == '__main__':
    main()
