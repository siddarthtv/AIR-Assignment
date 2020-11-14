#Import statements

import os
import pandas
import nltk
import numpy
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.metrics.pairwise import linear_kernel


#Pandas options
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.width', None)
pandas.set_option('display.max_colwidth', -1)


#Symbols
symbols = "!\"#$%&()*+-./:;<=>?@[\\]^_`{|}~\n"

#Stemmer
stemmer = nltk.stem.PorterStemmer()

#Path
path = os.getcwd()+'/AIR-Dataset'


def processQueryRanking(row):
    data = numpy.char.lower(row)
    for i in range(len(symbols)):
        data = numpy.char.replace(data, symbols[i], '')
    data = numpy.char.replace(data, ',', '')
    data = numpy.char.replace(data, "'", "")
    words = nltk.tokenize.word_tokenize(str(data))
    new_data = ""
    for w in words:
          new_data = new_data + " " + stemmer.stem(w)
    data = numpy.char.strip(new_data)
    return data

# create overall index  TO BE REMOVED AND REPLACED BY BLIST CODE
# index = {word1:{doc1:[row1, row2], doc2:[row2, row3]}, word2:{doc2:[row2]}} 
# docindex = {doc1:{row1:[pos1, pos2]}, doc2:{row1:[pos1], row2:[pos1, pos2]}}
# documents = {1:"BBCNEWS...csv", 2:"BBCNEWS...csv"}


def createFullIndex(column):
  index = dict()
  documents = dict()
  docindex = dict()
  doc = 0
  files_all = sorted(os.listdir(path))[1:]
  print(files_all)
  for filename in files_all:
    print("Document:", doc, filename)
    documents[doc] = filename
    df = pandas.read_csv(path+'/'+filename)
    r = 0
    temp = dict()
    for row in df[column]:
      pos = 0
      data = nltk.word_tokenize(str(process(row)))
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
  return index, documents, docindex




if __name__=="__main__":
  index, documents, docindex = createFullIndex("Snippet")



