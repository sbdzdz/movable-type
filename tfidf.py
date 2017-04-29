import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    pattern = './corpus/*.txt'
    files = glob.glob(pattern)
    vectorizer = TfidfVectorizer(input='filename', max_df=.9, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(files)

    similarity = cosine_similarity(tfidf_matrix)
    for index, file in enumerate(files):
        print("{}: {}".format(file, (list(similarity[index]))))

if __name__ == '__main__':
    main()
