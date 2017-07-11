from collections import Counter

d1 = 'egg bacon sausage and spam'
d2 = 'spam bacon sausage and spam'
d3 = 'spam egg spam spam bacon and spam'
d3 = 'spam egg spam spam bacon and spam'
d4 = 'spam spam spam spam spam spam baked beans spam spam spam'
D = {d1, d2, d3, d4}

counter = Counter(d2.split())
total = sum(counter.values())
term_frequencies = {term: counter[term]/total for term in counter}
