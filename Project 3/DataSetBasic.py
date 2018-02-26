#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:21:13 2018

@author: carsonluuu
"""
##Load data from files
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importing the dataset
#header = ['user_id', 'movie_id', 'rating', 'timestamp']
#dataset = pd.read_csv('ratings.csv', names=header)
#
#n_users = dataset.user_id.unique().shape[0]
#n_items = dataset.movie_id.unique().shape[0]

dataset = pd.read_csv('ratings.csv')
X = dataset.iloc[:, :-1].values
src_data = X[:,0:3]

user_total = int(max(X[:,0]))
movieID_total = int(max(X[:,1]))

R = np.empty((user_total, movieID_total))
R[:] = np.nan

for i in range(len(src_data)):
    m = int(src_data[i][0] - 1)
    n = int(src_data[i][1] - 1)
    R[m][n] = src_data[i][2]
    
#Compute the sparsity of the movie rating data
#print(1 - np.isnan(data_matrix).sum()/(user_total*movieID_total))

rates_count = []
ratings_range = np.arange(0, 5.5, 0.5)

for rate in ratings_range:
    rate_count = np.count_nonzero(R == rate)
    rates_count.append(rate_count)
    
plt.bar(ratings_range,rates_count,0.4,color="green", tick_label=np.arange(0, 5.5, 0.5)) 

movies_count = []
for movie in range(len(R.T)):
    movies_count.append(np.count_nonzero(~np.isnan(R[:,movie])))
movies_count.sort(reverse=True)
plt.plot(movies_count)
plt.title("Distribution of Ratings among Movies", fontsize = 10)
plt.ylabel("Rating Number")
plt.xlim([0,9125])
plt.xlabel("MovieID")

users_count = []
for user in range(len(R)):
    users_count.append(np.count_nonzero(~np.isnan(R[user,:])))
users_count.sort(reverse=True)

plt.plot(users_count)
plt.title("Distribution of Ratings among Users", fontsize = 10)
plt.ylabel("Rating Number")
plt.xlabel("UserID")
#
movies_var = []
for movie in range(len(R.T)):
    movies_var.append(np.nanvar(R[:,movie]))

count = np.zeros(10)
for var in movies_var:
    if (var >= 0 and var < 0.5):
        count[0]= count[0] + 1
    elif (var >= 0.5 and var < 1):
        count[1]= count[1] + 1
    elif (var >= 1 and var < 1.5):
        count[2]= count[2] + 1
    elif (var >= 1.5 and var < 2):
        count[3]= count[3] + 1
    elif (var >= 2 and var < 2.5):
        count[4]= count[4] + 1
    elif (var >= 2.5 and var < 3):
        count[5]= count[5] + 1
    elif (var >= 3 and var < 3.5):
        count[6]= count[6] + 1
    elif (var >= 3.5 and var < 4):
        count[7]= count[7] + 1
    elif (var >= 4 and var < 4.5):
        count[8]= count[8] + 1
    elif (var >= 4.5 and var < 5):
        count[9]= count[9] + 1

plt.bar(np.arange(0.25, 5.25, 0.5),count,0.48,color="purple")

#print('Finding the best k for KNeighbors classifier...')
#
#ks = np.arange(1, 51, 2)
#cross_val = []
#for k in ks:
#        knn_clf = KNeighborsClassifier(n_neighbors = k)
#        cross_val.append(1 - cross_val_score(knn_clf, X, y, cv=10, scoring='accuracy').mean());
#plt.plot(ks, cross_val, 'm-')
#plt.title('The Effect of k for KNN on Validation Error Under 10-Fold Cross Validation', fontsize=12)
#plt.xlabel('k Values', fontsize=10)
#plt.ylabel('10-Fold Cross Validation Average Error', fontsize=10)


