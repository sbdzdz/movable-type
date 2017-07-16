from collections import Counter
from math import log

doc1 = 'egg bacon sausage and spam'
doc2 = 'spam bacon sausage and spam'
doc3 = 'spam spam spam baked beans and spam'
corpus = {doc1, doc2, doc3}

count = Counter(doc1.split())
terms = count.keys()
total = sum(count.values())
tf_idf = {}

for term in terms:
    tf = count[term]/total
    idf = log(len(corpus)/(sum(term in document.split() for document in corpus)))
    tf_idf[term] = tf*idf

print(tf_idf)
