#import statements

import os
import pandas
import nltk
import numpy

#system requirement: python -m nltk.downloader 'punkt'

#pandas options

pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.width', None)
pandas.set_option('display.max_colwidth', -1)


#Symbols
symbols = "!\"#$%&()*+-./:;<=>?@[\\]^_`{|}~\n"

#Stemmer
stemmer = nltk.stem.PorterStemmer()

#Path variable
path = os.getcwd()+'/AIR-Dataset'

 

#Creating bigrams for wildcard characters
def wildcard(word, bigrams, st):
    word = "$"+word+"$"
    for i in range(len(word)-1):
        bi = word[i:i+2]
        if bi not in bigrams.keys():
            bigrams[bi] = sortedset()
            bigrams[bi].add(st)
        elif st not in bigrams[bi]:
            bigrams[bi].add(st)
    
    
    
    
  #Processing snippets before indexing
  def process(row, bigrams):
    data = numpy.char.lower(row)
    for i in range(len(symbols)):
        data = numpy.char.replace(data, symbols[i], '')
    data = numpy.char.replace(data, ',', '')
    data = numpy.char.replace(data, "'", "")
    words = nltk.tokenize.word_tokenize(str(data))
    new_data = ""
    for w in words:
        st = stemmer.stem(w)
        wildcard(w, bigrams, st)
        new_data = new_data + " " + st
    data = numpy.char.strip(new_data)
    return data
  
  
  #Creating different indices
  

# index = {word1:{doc1:[row1, row2], doc2:[row2, row3]}, word2:{doc2:[row2]}} 
# docindex = {doc1:{row1:[pos1, pos2]}, doc2:{row1:[pos1], row2:[pos1, pos2]}}
# documents = {1:"BBCNEWS...csv", 2:"BBCNEWS...csv"}


  def createFullIndex(column):
    index = dict()
    documents = dict()
    docindex = dict()
    bigrams = dict()
    doc = 0
    files_all = sorted(os.listdir(path))[1:]
    for filename in files_all:
        print("Document:", doc, filename)
        documents[doc] = filename
        df = pandas.read_csv(path+'/'+filename)
        r = 0
        temp = dict()
        for row in df[column]:
            pos = 0
            data = nltk.word_tokenize(str(process(row, bigrams)))
            for tok in data:
                if tok in index:
                #creating fullindex
                    if doc in index[tok]:
                        index[tok][doc].add(r)
                    else:
                        index[tok][doc] = set()
                        index[tok][doc].add(r)
                else:
                #creating fullindex
                    index[tok] = dict()
                    index[tok][doc] = set()
                    index[tok][doc].add(r)
                if tok in temp:
                    #creating docindex
                    if r in temp[tok].keys():
                        temp[tok][r].append(pos)
                    else:
                        temp[tok][r] = [pos]
                else:
                    #creating docindex
                    temp[tok] = dict()
                    temp[tok][r] = [pos]
                pos = pos+1
            r = r+1 
        docindex[doc] = temp
        doc = doc+1
    return index, documents, docindex, bigrams
  
  
  if __name__ == "__main__":
    index, documents, docindex, bigrams = createFullIndex("Snippet")
    
