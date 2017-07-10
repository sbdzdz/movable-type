from nltk.corpus import gutenberg

titles = gutenberg.fileids()
for title in titles:
    with open(title, 'w') as out:
        out.write(gutenberg.raw(title))
