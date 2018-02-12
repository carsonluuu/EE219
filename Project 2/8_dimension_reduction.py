# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 03:53:47 2018

@author: Chenguang
"""
from sklearn.datasets import fetch_20newsgroups
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction import text
import string
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pylab
from sklearn.preprocessing import Normalizer, PolynomialFeatures, FunctionTransformer

categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

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

def fetchDataAll_train(category_all):
    trainData = fetch_20newsgroups(subset='train', categories=category_all, shuffle=True, random_state=42)
    return trainData

def fetchDataAll_test(category_all):
    testData = fetch_20newsgroups(subset='test', categories=category_all, shuffle=True, random_state=42)
    return testData

train = fetchDataAll_train(categories)
test = fetchDataAll_test(categories)
train_data = train.data
train_target = train.target
test_data = test.data
test_target = test.target


train_data=textPreprocess(train_data)
test_data=textPreprocess(test_data)
data = TruncatedSVD(n_components=1000).fit_transform(train_tfidf)

count_vect = CountVectorizer(min_df=3, stop_words=stop_words)
train_counts = count_vect.fit_transform(train_data)
tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_counts)

for i in range(train_target.shape[0]):
    if train_target[i] >= 0 and train_target[i] <= 3:
        train_target[i] = 0
    else:
        train_target[i] = 1

### choose best r, svd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics
def find_r(k):
    r_data = data[:, :k]
    print(r_data.shape)
    km = KMeans(n_clusters=2, n_init=30)
    predicted = km.fit_predict(r_data)
    print("contingency matrix:")
    cm = metrics.confusion_matrix(train_target, predicted)
    print(cm)
    results = []
    results.append(metrics.homogeneity_score(train_target, predicted))
    results.append(metrics.completeness_score(train_target, predicted))
    results.append(metrics.v_measure_score(train_target, predicted))
    results.append(metrics.adjusted_rand_score(train_target, predicted))
    results.append(metrics.adjusted_mutual_info_score(train_target, predicted))
    return results

r_range = [1,2,3,5,10,20,50,100,300]
homogeneity_score = []
completeness_score = []
v_measure_score = []
adjusted_rand_score = []
adjusted_mutual_info_score = []
for k in r_range:
    results = find_r(k=k)
    homogeneity_score.append(results[0])
    completeness_score.append(results[1])
    v_measure_score.append(results[2])
    adjusted_rand_score.append(results[3])
    adjusted_mutual_info_score.append(results[4])

dimension_array = [1, 2, 3, 5, 10, 20, 50, 100, 300]

for d in dimension_array:
    svd = TruncatedSVD(n_components=d, n_iter=10, random_state=42)
#   normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd)
    X_reduced = lsa.fit_transform(X)
    homo, cmplt, rand, vmeasure, mutual = k_means(X_reduced, labels, 'truncatedSVD')
    
    h.append(homo)
    c.append(cmplt)
    v.append(vmeasure)
    r.append(rand)
    m.append(mutual)
    
plt.plot(dimension_array, h, color='r', label='homogeneity_score')
plt.plot(dimension_array, c, color='g', label='completeness_score')
plt.plot(dimension_array, v, color='m', label='v_measure_score')
plt.plot(dimension_array, r, color='y', label='adjusted_rand_score')
plt.plot(dimension_array, m, color='b', label='adjusted_mutual_info_score')
plt.legend(loc='best')
plt.xlabel('Dimension')
plt.ylabel('Scores')
plt.title('Clustering Performance vs Dimension for TruncatedSVD')
plt.show()
    

### choose best r, nmf
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics
def find_r(k):
    r_data = NMF(n_components=k).fit_transform(train_tfidf)
    km = KMeans(n_clusters=2, n_init=30)
    predicted = km.fit_predict(r_data)
    print("contingency matrix:")
    cm = metrics.confusion_matrix(train_target, predicted)
    print(cm)
    results = []
    results.append(metrics.homogeneity_score(train_target, predicted))
    results.append(metrics.completeness_score(train_target, predicted))
    results.append(metrics.v_measure_score(train_target, predicted))
    results.append(metrics.adjusted_rand_score(train_target, predicted))
    results.append(metrics.adjusted_mutual_info_score(train_target, predicted))
    return results

r_range = [1,2,3,5,10,20,50,100,300]
homogeneity_score = []
completeness_score = []
v_measure_score = []
adjusted_rand_score = []
adjusted_mutual_info_score = []
for k in r_range:
    results = find_r(k=k)
    homogeneity_score.append(results[0])
    completeness_score.append(results[1])
    v_measure_score.append(results[2])
    adjusted_rand_score.append(results[3])
    adjusted_mutual_info_score.append(results[4])

dimension_array = [1, 2, 3, 5, 10, 20, 50, 100, 300]

for d in dimension_array:
    nmf = NMF(n_components=d)
#   normalizer = Normalizer(copy=False)
    lsa = make_pipeline(nmf)
    X_reduced = lsa.fit_transform(X)
    homo, cmplt, rand, vmeasure, mutual = k_means(X_reduced, labels, 'NMF')
    
    h.append(homo)
    c.append(cmplt)
    v.append(vmeasure)
    r.append(rand)
    m.append(mutual)
    
plt.plot(dimension_array, h, color='r', label='homogeneity_score')
plt.plot(dimension_array, c, color='g', label='completeness_score')
plt.plot(dimension_array, v, color='m', label='v_measure_score')
plt.plot(dimension_array, r, color='y', label='adjusted_rand_score')
plt.plot(dimension_array, m, color='b', label='adjusted_mutual_info_score')
plt.legend(loc='best')
plt.xlabel('Dimension')
plt.ylabel('Scores')
plt.title('Clustering Performance vs Dimension for NMF')
plt.show()