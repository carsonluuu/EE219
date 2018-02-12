#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:12:11 2018

@author: carsonluuu
"""
import logging

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text #stop words
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

from sklearn.decomposition import TruncatedSVD, NMF

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, PolynomialFeatures, FunctionTransformer

def calculate_stats(target, predicted):
    homogeneity = metrics.homogeneity_score(target, predicted)
    completeness = metrics.completeness_score(target, predicted)
    v_measure_score = metrics.v_measure_score(target, predicted)
    adjusted_Rand_Index = metrics.adjusted_rand_score(target, predicted)
    adjusted_Mutual_Info_Score = metrics.adjusted_mutual_info_score(target, predicted)
    return (homogeneity, completeness, v_measure_score, adjusted_Rand_Index, adjusted_Mutual_Info_Score)

def fetchDataAll(category_all):
    trainData = fetch_20newsgroups(subset='all', categories=category_all, shuffle=True, random_state=42)
    return trainData

if __name__ == "__main__":
    categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',\
                  'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles',\
                  'rec.sport.baseball', 'rec.sport.hockey']
    trainData = fetchDataAll(categories)
    stop_words = text.ENGLISH_STOP_WORDS
    
    vectorizer = CountVectorizer(analyzer='word', stop_words=stop_words, max_df=0.99, min_df=3)
    dataTrainCounts = vectorizer.fit_transform(trainData.data)
    print(len(vectorizer.get_feature_names()))

    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(dataTrainCounts)
    print(tfidf.shape)
    predicted = KMeans(n_clusters=2, random_state=0).fit(tfidf) # applying k-means
    labels = trainData.target//4 #8 divide 4 is 2
    print("Confusion Matrix is ", metrics.confusion_matrix(labels, predicted.labels_))
    print(calculate_stats(labels, predicted.labels_)) #get the given 5 parameters

#    svd = TruncatedSVD(n_components=98)
#    normalizer = Normalizer(copy=False)
#    lsa = make_pipeline(svd, normalizer)
#    X_lsa = lsa.fit_transform(tfidf)
#    predicted = KMeans(n_clusters=2, random_state=0).fit(X_lsa) # applying k-means
#    labels = trainData.target//4 #8 divide 4 is 2
#    print("Confusion Matrix is ", metrics.confusion_matrix(labels, predicted.labels_))
#    print(calculate_stats(labels, predicted.labels_)) #get the given 5 parameters
#
#    variance_retained = svd.explained_variance_ratio_
#    sigu = svd.singular_values_
    