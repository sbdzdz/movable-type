from collections import Counter
from math import log

doc1 = 'egg bacon sausage and spam'
doc2 = 'spam bacon sausage and spam'
doc3 = 'spam egg spam spam bacon and spam'
doc4 = 'spam spam baked beans and spam'
corpus = {doc1, doc2, doc3, doc4}

count = Counter(doc2.split())
terms = count.keys()
tf_idf = {}

for term in terms:
    tf = count[term]/len(terms)
    idf = log(len(corpus)/(sum(term in document.split() for document in corpus)))
    tf_idf[term] = tf*idf

print(tf_idf)
