# Movable type
Playing around with texts from Project Gutenberg:
1. Download a couple of books
2. Map each book to a multi-dimensional vector space using the tf-idf weights
3. Find distinct terms for each book
4. Cluster using k-means
5. Visualize using PCA and t-SNE

### tfidf_example.py
A tiny example of calculating tf-idf on a mock document collection.

### fetch_books.py
A short helper script to fetch the texts of a couple of books from Project Gutenberg and save them in separate files.

### analysis.py
The actual analysis.
