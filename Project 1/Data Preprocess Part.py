#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 19:45:31 2018

@author: Jiahui Lu
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text #stop words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from sklearn.decomposition import NMF

def fetchDataAll(category_all):
    trainData = fetch_20newsgroups(subset='train', categories=category_all, shuffle=True, random_state=42)
    return trainData

def fetchData(category):
    trainData = fetch_20newsgroups(subset='train', categories=[category], shuffle=True, random_state=42)
    return trainData

def draw(trainData):
    frequencyCount = pd.value_counts(trainData.target, sort=False)
    targetNames = pd.DataFrame(trainData.target_names)
    targetNames.columns = ["name"]
    targetVal = pd.DataFrame(frequencyCount)
    targetVal.columns = ["counts"]
    
    targetMerge = targetNames.join(targetVal)
    
    plt.figure(figsize=(8, 6), dpi=100, facecolor='w')
    rects = plt.bar(np.arange(8),targetMerge["counts"], color='green')
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2., 1.03*height, '%s' % int(height))
    plt.title("Number of Traning Documents in Each Class")
    plt.xticks(np.arange(len(trainData.target_names)), targetMerge["name"], rotation=25)
    plt.show()
 

def textPreprocess(trainData):
    #import nltk
    #nltk.download('wordnet')     
    from nltk.tokenize import RegexpTokenizer #remove pun
    tokenizer = RegexpTokenizer(r'\w+')
    from nltk.stem.wordnet import WordNetLemmatizer
    lmtzr = WordNetLemmatizer()
    #Lemmatizer eg. going -> go(v); nicer -> nice(a); cars -> car(n) 
    trainData = [tokenizer.tokenize(sentence) for sentence in trainData]
    trainData = [" ".join(sentence) for sentence in trainData]
    trainData = [[lmtzr.lemmatize(word, 'n') for word in sentence.split(" ")] for sentence in trainData]
    trainData = [" ".join(sentence) for sentence in trainData]
    trainData = [[lmtzr.lemmatize(word, 'v') for word in sentence.split(" ")] for sentence in trainData] 
    trainData = [" ".join(sentence) for sentence in trainData]
    trainData = [[lmtzr.lemmatize(word, 'a') for word in sentence.split(" ")] for sentence in trainData] 
    trainData = [" ".join(sentence) for sentence in trainData]        
    return trainData
    
def TFxIDF_Length(trainData, minimumDf):
    # return documnt frequency
    stop_words = text.ENGLISH_STOP_WORDS
    vectorizer = CountVectorizer(min_df=minimumDf, stop_words=stop_words)
    dataTrainCounts = vectorizer.fit_transform(trainData)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(dataTrainCounts)
    return(tfidf.shape[1]) 

def TFxIDF(trainData, minimumDf):
    # return documnt frequency
    stop_words = text.ENGLISH_STOP_WORDS
    vectorizer = CountVectorizer(min_df=minimumDf, stop_words=stop_words)
    dataTrainCounts = vectorizer.fit_transform(trainData)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(dataTrainCounts)
    return tfidf

def TFxICF():
    train20 = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
    class_text = []
    for i in range(20):
        class_text.append("")
    for i in range(len(train20.data)):
        category = train20.target[i]
        class_text[category] = class_text[category] + " " + train20.data[i]
    #class_text = textPreprocess(class_text)
    
    stop_words = text.ENGLISH_STOP_WORDS
    vectorizer = CountVectorizer(stop_words=stop_words, min_df = 5, max_df=0.6)
    train20_counts = vectorizer.fit_transform(class_text)
    transformer = TfidfTransformer()
    train20_tfidf = transformer.fit_transform(train20_counts)
    classes = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',\
               'misc.forsale', 'soc.religion.christian']
    for i in range(20):
        for j in range(4):
            if train20.target_names[i] == classes[j]:
                curClass = train20_tfidf.toarray()[i]
                top10Index = sorted(range(len(curClass)), key=lambda index: curClass[index])[-10:]
                top10Term = []
                for index in top10Index:
                    top10Term.append(vectorizer.get_feature_names()[index])
                print("top ten terms for " + classes[j] + " :")
                print(top10Term)
 
def lsa(data):    
    svd = TruncatedSVD(n_components=50, n_iter=10,random_state=42)
    train_data_lsa = svd.fit_transform(data)
    return train_data_lsa

def nmf(data):
    model = NMF(n_components=50, init='random', random_state=0)
    train_data_tfidf = TFxIDF(data, 5)
    train_data_NMF = model.fit_transform(train_data_tfidf)
    
    return train_data_NMF
  
if __name__ == "__main__":
    categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',\
                  'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles',\
                  'rec.sport.baseball', 'rec.sport.hockey']
    eightClass_train = fetchDataAll(categories)
    
    #draw(eightClass_train)
#    dfMin2 = []
#    dfMin5 = []
#    for i in range(len(categories)):
#        importData = fetchData(categories[i])
#        importData = textPreprocess(importData.data)
#        min2 = TFxIDF_Length(importData,2)
#        dfMin2.append(min2)
#        min5 = TFxIDF_Length(importData,5)
#        dfMin5.append(min5)
#        print("finish " + str(i) + " --------")
#    print(dfMin2)
#    print(dfMin5)    
    
    TFxICF()

###############################################################################
#  Use LSA to vectorize the articles.
###############################################################################
#    for i in range(1):
#        importData = fetchData(categories[i])
#        importData = textPreprocess(importData.data)
#        lsa = lsa(importData)
#        nmf = nmf(importData)

    

    
    