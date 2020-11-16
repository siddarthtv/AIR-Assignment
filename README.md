# Search Engine

This repository contains the code for a basic search engine as part of our Algorithms for Information Retrieval Assignment. 

## Flow

- Posting list and bigrams posting list are created from the Snippets (run createIndices.py)
- These are used to narrow down candidate documents for word queries, phrase queries, wilcard queries
- Vector space models of the candidates are built with TF-IDF scores
- Candidate documents are ranked based on relevance and displayed

## Dataset

AIR-Dataset: Contains 418 CSV files, with approximately 95k rows.

## Code

createIndices.py: Program to create inverted index and postings list and save to file
searchEngine.py: Program to load index from file and functions to search given a query
server.py: Program to provide user interface for queries
templates/index.html: HTML file for query page
templates/results.html: HTML file for results page

## Steps to run

1. Run python3 createIndices.py to create the different indexes.
2. Run python3 server.py to launch the flask server (for user interface).
3. On a web browser, open http://localhost:5000 to open the query page.
4. Use the user interface to perform queries.
