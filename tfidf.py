import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from nltk.corpus import gutenberg

corpus = {title: gutenberg.raw(title) for title in gutenberg.fileids()}
vectorizer = TfidfVectorizer(max_df=.9, stop_words='english')

weights = vectorizer.fit_transform(corpus.values())
weights = weights.toarray()
#weights = np.asarray(weights.mean(axis=0)).ravel().tolist()
#weights_df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'weight': weights})

for index, title in enumerate(corpus):
    print(title)
    df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'weight': weights[index]})
    print(df.sort_values(by='weight', ascending=False).head(5))
    print("\n\n\n")
