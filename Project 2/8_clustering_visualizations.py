# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 22:13:31 2018

@author: Chenguang
"""

from sklearn.datasets import fetch_20newsgroups
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import normalize
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction import text
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline


categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
dataset = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)


vectorizer = TfidfVectorizer(min_df=3, stop_words='english', use_idf=True)
X = vectorizer.fit_transform(dataset.data)

#for i in range(train_target.shape[0]):
#    if train_target[i] >= 0 and train_target[i] <= 3:
#        train_target[i] = 0
#    else:
#        train_target[i] = 1
labels = dataset.target//4

def plot_clusters(actual_labels, clustered_labels, X_2d, centers, reducer):
    color = ["r", "g"]
    mark = ["o", "+"]
    for i in range(len(labels)):
        plt.scatter(X_2d[i, 0], X_2d[i, 1], s=12, marker=mark[actual_labels[i]], color=color[clustered_labels[i]], alpha=0.5)
    for i in range(2):
        plt.scatter(centers[i, 0], centers[i, 1], marker='^', s=100, linewidths=5, color='k', alpha=0.6)
    plt.title('Clustering results with ' + reducer)
    plt.show()

#=================================SVD results=================================
svd = TruncatedSVD(n_components=2)
best_tfidf_svd = svd.fit_transform(X)
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=1)
kmeans.fit(best_tfidf_svd)
clustered_labels = kmeans.labels_
centers = kmeans.cluster_centers_
plot_clusters(labels, clustered_labels, best_tfidf_svd, centers, 'TruncatedSVD')


# =================================MNF results=================================
nmf = NMF(n_components=3, init='random', random_state=42)
best_tfidf_nmf = nmf.fit_transform(X)
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=1)
kmeans.fit(best_tfidf_nmf)
clustered_labels = kmeans.labels_
centers = kmeans.cluster_centers_
plot_clusters(labels, clustered_labels, best_tfidf_nmf, centers, 'NMF')

#=================================SVD results=================================
#svd = TruncatedSVD(n_components=2)
#normalizer = Normalizer(copy=False)
#lsa = make_pipeline(svd, normalizer)
#X_reduced = lsa.fit_transform(X)
#kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=1)
#kmeans.fit(best_tfidf_svd)
#clustered_labels = kmeans.labels_
#centers = kmeans.cluster_centers_
#plot_clusters(labels, clustered_labels, X_reduced, centers, 'TruncatedSVD')


# =================================MNF results=================================
nmf = NMF(n_components=4, init='random', random_state=42)
best_tfidf_nmf = nmf.fit_transform(X)
normalizer = Normalizer(copy=False)
nmff = make_pipeline(svd, normalizer)
X_reduced = nmff.fit_transform(X)
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=1)
kmeans.fit(best_tfidf_nmf)
clustered_labels = kmeans.labels_
centers = kmeans.cluster_centers_
plot_clusters(labels, clustered_labels, X_reduced, centers, 'NMF')


# =================================Transformation=================================



### Apply logarithmic transformation after NMF
logtransformer = FunctionTransformer(np.log1p)
best_tfidf_nmf_log = logtransformer.transform(best_tfidf_nmf)
kmeans = KMeans(n_clusters=2, n_init=30)
kmeans.fit_predict(best_tfidf_nmf_log)
clustered_labels = kmeans.labels_
centers = kmeans.cluster_centers_
plot_clusters(labels, clustered_labels, X_reduced, centers, 'NMF')


### Apply norm + log after NMF
best_tfidf_nmf_norm = normalize(best_tfidf_nmf)
best_tfidf_nmf_norm_log = logtransformer.transform(best_tfidf_nmf_norm)
kmeans = KMeans(n_clusters=2, n_init=30)
km = kmeans.fit_predict(best_tfidf_nmf_norm_log)
fig = plt.figure()
plt.scatter(best_tfidf_nmf_norm_log[:,0], best_tfidf_nmf_norm_log[:,1], c = km, s=12, alpha=0.5)
plt.show()
print("contingency matrix:")
print(metrics.confusion_matrix(train_target, kmeans.labels_))

### Apply log + norm after NMF
best_tfidf_nmf_log = logtransformer.transform(best_tfidf_nmf)
best_tfidf_nmf_log_norm = normalize(best_tfidf_nmf_log)
kmeans = KMeans(n_clusters=2, n_init=30)
km = kmeans.fit_predict(best_tfidf_nmf_log_norm)
fig = plt.figure()
plt.scatter(best_tfidf_nmf_log_norm[:,0], best_tfidf_nmf_log_norm[:,1], c = km, s=12, alpha=0.5)
plt.show()
print("contingency matrix:")
print(metrics.confusion_matrix(train_target, kmeans.labels_))

