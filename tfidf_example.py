from collections import Counter
from math import log

doc1 = 'egg bacon sausage and spam'
doc2 = 'spam bacon sausage and spam'
doc3 = 'spam spam spam spam spam spam baked beans spam spam spam and spam'
D = {doc1, doc2, doc3}

count = Counter(doc2.split())
terms = count.keys()
total = sum(count.values())

tf = {t: count[t]/total for t in terms}
idf = {t: log(len(D)/sum(t in d.split() for d in D)) for t in terms}
tf_idf = {t: round(tf[t]*idf[t], 3) for t in terms}
print(tf_idf)
