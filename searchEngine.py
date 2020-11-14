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

#Processing queries prior to ranking
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
      data = nltk.word_tokenize(str(processQueryRanking(row)))
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


####### HANDLING DIFFERENT TYPES OF QUERIES #############

def singleWordQuery(query):
    cleaned = str(processQueryRanking(query))
    ans = index.get(cleaned)
    return ans


######## RANKING #########

#Checks query length and calls the functions accordingly
def handleQuery(query):
    if len(query.split())==1: #single word
        return singleWordQuery(query)
    

#Creates DF using candidate docs(row)
def corpusdf(ans):
    ensuing_df_data=[]

    for file in ans:
        with open(path+"/"+documents[file]) as csv_file:
            csv_reader = csv.reader(csv_file)
            rows = list(csv_reader)
    
            for each_row in ans[file]:
                ensuing_df_data.append(rows[each_row+1])       
    ensuing_df = pandas.DataFrame(ensuing_df_data, columns = ['URL', 'MatchDateTime', 'Station', 'Show', 'IAShowID', 'IAPreviewThumb', 'Snippet'])
    return ensuing_df

#Creates df of the query and concatenates this with corpus_df
def querydf(query):
    data_query = [['filler', 'values', 'for', 'the', 'query','data',query]] 
    querydf = pandas.DataFrame(data_query, columns = ['URL', 'MatchDateTime', 'Station', 'Show', 'IAShowID', 'IAPreviewThumb', 'Snippet']) 
    global corpus_df
    to_concatenate = [corpus_df, querydf]
    final_df = pandas.concat(to_concatenate)
    return final_df


#Creates tf-idf vector space using concatenated df
def vectorSpaceModel():   
    tfidfv = TfidfVectorizer(min_df = 1, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range={1,3}, use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english')
    global final_df
    query_tfv_mat=tfidfv.fit_transform(final_df['Snippet'])
    return query_tfv_mat

#Finds similarity scores (cosine), sorts the docs based on sim scores and returns top 22 results
def rankingFunc(query_tfv_mat):
    global final_df
    sim_scores = linear_kernel(query_tfv_mat, query_tfv_mat[len(final_df)-1]).flatten()
    relevant_docs_indices = sim_scores.argsort()[:-24:-1]
    return relevant_docs_indices
    
    
#Displays the results in a given format
def displayRanked(ranked_docs):
    global final_df
    rank = 1
    rank_list = numpy.delete(ranked_docs,0)
    for i in rank_list:
        print('Rank: ',rank)
        print(final_df.iloc[i])
        print()
        print('\n\n')
        rank+=1
    print(rank-1,' results found.')


#Main function
if __name__=="__main__":
    index, documents, docindex = createFullIndex("Snippet")
    query = 'timberlake'
    ans=handleQuery(query)
    corpus_df = corpusdf(ans)
    final_df = querydf(query)
    query_tfv_mat = vectorSpaceModel()
    ranked_docs = rankingFunc(query_tfv_mat)
    displayRanked(ranked_docs)





