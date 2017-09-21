mkdir books

# move all zip files to a single directory
find ./aleph.gutenberg.org -name "*.zip" -exec mv {} books \;

# unzip all
find books -type f -name "*.zip" -execdir unzip {} \;

# find all books for each author from the list and move them to a separate directory
while read p; do
    echo $p
    mkdir -p authors/${p##* }
    rg -lm 1 "Author: $p" data/books | xargs -I {} cp {} authors/${p##* }
done < author_list
