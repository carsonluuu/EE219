#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 14:35:21 2018

@author: carsonluuu
"""

from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

from math import log
import matplotlib.pyplot as plt


def k_means(n, X_reduced, labels, dim_reduce):
    # =============================K-Means Clustering===============================
    km = KMeans(n_clusters=n, init='k-means++', max_iter=100, n_init=1)
    km.fit(X_reduced)
    
    homo = metrics.homogeneity_score(labels, km.labels_)
    complete = metrics.completeness_score(labels, km.labels_)
    vmeasure = metrics.v_measure_score(labels, km.labels_)
    rand = metrics.adjusted_rand_score(labels, km.labels_)
    mutual = metrics.adjusted_mutual_info_score(labels, km.labels_)
    
    return homo, complete, vmeasure, rand, mutual

if __name__ == "__main__":
    comp_tech_subclasses = ['comp.graphics',
                        'comp.os.ms-windows.misc',
                        'comp.sys.ibm.pc.hardware',
                        'comp.sys.mac.hardware',
                        'comp.windows.x']
    rec_act_subclasses = ['rec.autos',
                          'rec.motorcycles',
                          'rec.sport.baseball',
                          'rec.sport.hockey']
    science_subclass = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space']
    miscellaneous_subclass = ['misc.forsale']
    politics_subclass = ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast']
    religion_subclass = ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian']
    
    dataset = fetch_20newsgroups(subset='all',
                                 categories=comp_tech_subclasses + rec_act_subclasses + science_subclass +
                                            miscellaneous_subclass + politics_subclass + religion_subclass,
                                 shuffle=True,
                                 random_state=42)
    labels = dataset.target
    
    vectorizer = TfidfVectorizer(min_df=3, stop_words='english', use_idf=True)
    X = vectorizer.fit_transform(dataset.data)
    labels = [1] * len(dataset.data)
#    for i in range(len(dataset.data)):
#        if dataset.target[i] <= 4:
#            labels[i] = 0
#        elif dataset.target[i] <= 8:
#            labels[i] = 1
#        elif dataset.target[i] <= 12:
#            labels[i] = 2
#        elif dataset.target[i] <= 13:
#            labels[i] = 3
#        elif dataset.target[i] <= 16:
#            labels[i] = 4
#        elif dataset.target[i] <= 19:
#            labels[i] = 5
#    svd = TruncatedSVD(n_components=50, n_iter=10, random_state=42)
#    nmf = NMF(n_components=50, random_state=42)
#
#    normalizer = Normalizer(copy=False)
#    lsa = make_pipeline(svd, normalizer)
#    nmf = make_pipeline(nmf, normalizer)
##    X_reduced = lsa.fit_transform(X)
#    X_reduced = nmf.fit_transform(X)
#    print(k_means(6, X_reduced, labels, 'svd'))
    
#    dimension_array = [213, 192, 182, 168, 134, 111, 95, 71, 50, 33, 20, 15, 10, 8, 7, 6, 5, 4, 3, 2]
# # dimension_array = range(0, 1) #for sweep
#    h = []
#    c = []
#    v = []
#    r = []
#    m = []
#    
#    for d in dimension_array:
#        svd = TruncatedSVD(n_components=d, n_iter=10, random_state=42)
#        normalizer = Normalizer(copy=False)
#        lsa = make_pipeline(svd, normalizer)
#        X_reduced = lsa.fit_transform(X)
#        homo, cmplt, rand, vmeasure, mutual = k_means(6, X_reduced, labels, 'truncatedSVD')
#    
#        h.append(homo)
#        c.append(cmplt)
#        v.append(vmeasure)
#        r.append(rand)
#        m.append(mutual)
#    
#    plt.plot(dimension_array, h, color='r', label='homogeneity_score')
#    plt.plot(dimension_array, c, color='g', label='completeness_score')
#    plt.plot(dimension_array, v, color='m', label='v_measure_score')
#    plt.plot(dimension_array, r, color='y', label='adjusted_rand_score')
#    plt.plot(dimension_array, m, color='b', label='adjusted_mutual_info_score')
#    plt.legend(loc='best')
#    plt.xlabel('Dimension')
#    plt.ylabel('Scores')
#    plt.title('Clustering Performance vs Dimension for TruncatedSVD')
#    plt.show()
    dimension_array = [67, 53, 44, 33, 30, 25, 22, 15, 10, 5, 3, 2]
    h2 = []
    c2 = []
    v2 = []
    r2 = []
    m2 = []
    
    h3 = []
    c3 = []
    v3 = []
    r3 = []
    m3 = []
    
    for d in dimension_array:
        print("obntaining for...")
        nmf = NMF(n_components=d, random_state=42)
        lsa = make_pipeline(nmf)
        X_reduced = lsa.fit_transform(X)
        homo, cmplt, vmeasure, rand, mutual = k_means(6, X_reduced, labels, 'NMF')    
        h2.append(homo)
        c2.append(cmplt)
        v2.append(vmeasure)
        r2.append(rand)
        m2.append(mutual)
        
        for j in range(X_reduced.shape[0]):
            for k in range(X_reduced.shape[1]):
                if X_reduced[j][k] == 0:
                    X_reduced[j][k] = -3.08
                else:
                    X_reduced[j][k] = log(X_reduced[j][k], 10)
        homo, cmplt, vmeasure, rand, mutual = k_means(6, X_reduced, labels, 'NMF_Log')    
        h3.append(homo)
        c3.append(cmplt)
        v3.append(vmeasure)
        r3.append(rand)
        m3.append(mutual)
    
    plt.plot(dimension_array, h2, color='r', label='homogeneity_score')
    plt.plot(dimension_array, c2, color='g', label='completeness_score')
    plt.plot(dimension_array, v2, color='m', label='v_measure_score')
    plt.plot(dimension_array, r2, color='y', label='adjusted_rand_score')
    plt.plot(dimension_array, m2, color='b', label='adjusted_mutual_info_score')
    plt.plot(dimension_array, h3, 'r--', label='homogeneity_score LOG')
    plt.plot(dimension_array, c3, 'g--', label='completeness_score LOG')
    plt.plot(dimension_array, v3, 'm--', label='v_measure_score')
    plt.plot(dimension_array, r3, 'y--', label='adjusted_rand_score LOG')
    plt.plot(dimension_array, m3, 'b--', label='adjusted_mutual_info_score LOG')
    plt.legend(loc='best')
    plt.xlabel('Dimension')
    plt.ylabel('Scores')
    plt.title('Clustering performance vs Dimension by NMF')
    plt.show()


