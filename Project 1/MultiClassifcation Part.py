#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 17:08:14 2018

@author: carsonluuu
"""
import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc
from nltk.stem.snowball import SnowballStemmer
import string


categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']

def fetchDataAll(category_all):
    trainData = fetch_20newsgroups(subset='train', categories=category_all, shuffle=True, random_state=42)
    return trainData

train = fetchDataAll(categories)
test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

train_data = train.data
train_target = train.target
test_data = test.data
test_target = test.target

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


train_data = textPreprocess(train_data)
test_data = textPreprocess(test_data)



Bayes_clf = Pipeline([('vect', CountVectorizer(min_df=2, stop_words = text.ENGLISH_STOP_WORDS)),
                      ('tfidf', TfidfTransformer()),
                      ('nmf', NMF(n_components=50, random_state=42)),
                      ('clf', GaussianNB()),
])
    
OvO_SVM_clf = Pipeline([('vect', CountVectorizer(min_df=2, stop_words = text.ENGLISH_STOP_WORDS)),
                        ('tfidf', TfidfTransformer()),
                        ('nmf', NMF(n_components=50, random_state=42)),
                        ('clf', OneVsOneClassifier(LinearSVC(random_state=0))),
])

OvR_SVM_clf = Pipeline([('vect', CountVectorizer(min_df=2, stop_words = text.ENGLISH_STOP_WORDS)),
                        ('tfidf', TfidfTransformer()),
                        ('nmf', NMF(n_components=50, random_state=42)),
                        ('clf', OneVsRestClassifier(LinearSVC(random_state=0))),
])

Bayes_clf.fit(train_data, train_target)
Bayes_predicted = Bayes_clf.predict(test_data)
Bayes_cm = confusion_matrix(test_target, Bayes_predicted)
Bayes_accuracy = accuracy_score(test_target, Bayes_predicted)
Bayes_precision = precision_score(test_target, Bayes_predicted, average = 'weighted')
Bayes_recall = recall_score(test_target, Bayes_predicted, average = 'weighted')

print("Result for Naive Bayes Classifier")
print("confusion_matrix: ")
print(Bayes_cm)
print("accuracy = %f, precision = %f, recall＝ %f" %(Bayes_accuracy, Bayes_precision, Bayes_recall))    

OvO_SVM_clf.fit(train_data, train_target)
OvO_SVM_predicted = OvO_SVM_clf.predict(test_data)
OvO_SVM_cm = confusion_matrix(test_target, OvO_SVM_predicted)
OvO_SVM_accuracy = accuracy_score(test_target, OvO_SVM_predicted)
OvO_SVM_precision = precision_score(test_target, OvO_SVM_predicted, average = 'weighted')
OvO_SVM_recall = recall_score(test_target, OvO_SVM_predicted, average = 'weighted')

print("Result for One VS One SVM Classifier")
print("confusion_matrix: ")
print(Bayes_cm)
print("accuracy = %f, precision = %f, recall＝ %f" %(OvO_SVM_accuracy, OvO_SVM_precision, OvO_SVM_recall))

OvR_SVM_clf.fit(train_data, train_target)
OvR_SVM_predicted = OvR_SVM_clf.predict(test_data)
OvR_SVM_cm = confusion_matrix(test_target, OvR_SVM_predicted)
OvR_SVM_accuracy = accuracy_score(test_target, OvR_SVM_predicted)
OvR_SVM_precision = precision_score(test_target, OvR_SVM_predicted, average = 'weighted')
OvR_SVM_recall = recall_score(test_target, OvR_SVM_predicted, average = 'weighted')
print("Result for One VS Rest SVM Classifier")
print("confusion_matrix: ")
print(Bayes_cm)
print("accuracy = %f, precision = %f, recall＝ %f" %(OvR_SVM_accuracy, OvR_SVM_precision, OvR_SVM_recall))