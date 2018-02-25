#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:24:01 2018

@author: carsonluuu
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from surprise import KNNWithMeans
from surprise import accuracy
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc, roc_auc_score

def knn_measurement(data):
    
    mae_knn = []
    rmse_knn = []
    #sweeping 
    for kk in np.arange(2, 102, 2) :
        algo = KNNWithMeans(k=kk)
        temp = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10, verbose=True)
        #cross_validate ---> built-in method, cv is 10 here
        #temp is a dict type
        mae_knn.append(temp['test_mae'].mean())
        rmse_knn.append(temp['test_rmse'].mean())
    return mae_knn, rmse_knn

def trimming_knn_plot(data) :
    
    trainset, testset = train_test_split(data, test_size=.1)
    
    mae_knn = []
    rmse_knn = []
    #sweeping 
    for kk in np.arange(2, 102, 2) :
        algo = KNNWithMeans(k=kk)
        algo.fit(trainset)
        predictions = algo.test(testset)
        rmse_knn.append(accuracy.rmse(predictions))
        mae_knn.append(accuracy.mae(predictions))
    
    plt.figure(1)
    plt.plot(np.arange(2, 102, 2), rmse_knn)
    plt.title("Performance Evaluations Using Trimming Data for Average RMSE over k", fontsize = 10)
    plt.ylabel("Average MAE (testset)")
    plt.xlabel("k")
    plt.figure(2)
    plt.plot(np.arange(2, 102, 2), mae_knn)
    plt.title("Performance Evaluations Using 10-fold Trimming Data for Average MAE over k", fontsize = 10)
    plt.ylabel("Average MAE (testset)")
    plt.xlabel("k")

def binary_value(data, threshold) :
    trainset, testset = train_test_split(data, test_size=.1)
    
    algo = KNNWithMeans(k = 30)
    algo.fit(trainset)
    predictions = algo.test(testset)
    
    like0 = []#real
    like  = []#predict
    for row in range(len(predictions)) :
        like.append( 1 if predictions[row][3] > threshold else 0)
        like0.append(1 if predictions[row][2] > threshold else 0)
    #predictions[row][3] -> predict value
    #predictions[row][2] -> real value
    return like0, like

def ROC_curve(TS) :
    #----------------------------------
    #Ts is the value [2.5, 3, 3.5, 4]
    #----------------------------------
    like02dot5, like2dot5 = binary_value(data, TS)     
    fpr, tpr, thresholds = roc_curve(like02dot5, like2dot5)
    auroc = auc(fpr, tpr)
    print(roc_auc_score(like02dot5, like2dot5))
    plt.plot(fpr, tpr, label='ROC curve (auroc = %0.3f)' % auroc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('title..')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    dataset = pd.read_csv('ratings.csv')
    reader = Reader()
    data = Dataset.load_from_df(dataset, reader)
    
    
    ### ========== PLOT TO FIND BEST K ========== ###
#    mae_knn, rmse_knn = knn_measurement(data)
#    plt.plot(np.arange(2, 102, 2), mae_knn)
#    plt.title("Performance Evaluations Using 10-fold Cross Validation for Average MAE over k", fontsize = 10)
#    plt.ylabel("Average MAE (testset)")
#    plt.xlabel("k")
#    
#    plt.plot(np.arange(2, 102, 2), rmse_knn)
#    plt.title("Performance Evaluations Using 10-fold Cross Validation for Average RMSE over k", fontsize = 10)
#    plt.ylabel("Average RMSE (testset)")
#    plt.xlabel("k")    
    ### ================== END ================== ###
    
    
    #### ========== TRIMMING ========== ###
#    dataset_unpopular = dataset.groupby("movieId").filter(lambda x: len(x) <= 2)
#    dataset_popular   = dataset.groupby("movieId").filter(lambda x: len(x) > 2)
#    
#    data_unpopular = Dataset.load_from_df(dataset_unpopular, reader)
#    data_popular   = Dataset.load_from_df(dataset_popular, reader)
#    
#    trimming_knn_plot(data_popular)
#    trimming_knn_plot(data_unpopular)
#    
#    #variance triming UNDONE.........
    ### ============= END ============ ###


    ### ========== ROC ========== ###
#    ROC_curve(2.5)  #<----- NOT SURE STRANGE...
    ### ========== END ========== ###
    
    