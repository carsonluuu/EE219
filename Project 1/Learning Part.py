# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 17:26:10 2018

@author: Chenguang
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
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc
from nltk.stem.snowball import SnowballStemmer
import string
from sklearn.linear_model import LogisticRegression


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

for i in range(train_target.shape[0]):
    if train_target[i] >= 0 and train_target[i] <= 3:
        train_target[i] = 0
    else:
        train_target[i] = 1
# classify the test_target based on class, instead of subclasses
for i in range(test_target.shape[0]):
    if test_target[i] >= 0 and test_target[i] <= 3:
        test_target[i] = 0
    else:
        test_target[i] = 1


##--------Hard Margin SVC using LSI----------------        
hard_svc_clf = Pipeline([('vect', CountVectorizer(min_df=5, stop_words=text.ENGLISH_STOP_WORDS)),
                         ('tfidf', TfidfTransformer()),
                         ('svd', TruncatedSVD(n_components=50, n_iter=10, random_state=42)),
                         ('clf', SVC(C = 1000, kernel='linear', probability=True)),
])

hard_svc_clf = hard_svc_clf.fit(train_data, train_target)

predicted_prob = hard_svc_clf.predict_proba(test_data)[:, 1]
predicted = hard_svc_clf.predict(test_data)

fpr, tpr, thresholds = roc_curve(test_target, predicted_prob)
auroc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (auroc = %0.3f)' % auroc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve--Hard Margin SVC using LSI')
plt.legend(loc="lower right")
plt.show()

cm = confusion_matrix(test_target, predicted)
accuracy = accuracy_score(test_target, predicted)
precision = precision_score(test_target, predicted, pos_label = 1)
recall = recall_score(test_target, predicted, pos_label = 1)
print("OC Curve for Hard Margin SVM Classifier")
print("confusion_matrix: ")
print(cm)
print("accuracy = %f, precision = %f, recall＝ %f" %(accuracy, precision, recall))


#--------Hard Margin SVC using NMF----------------
hard_svc_clf_nmf = Pipeline([('vect', CountVectorizer(min_df=5, stop_words=text.ENGLISH_STOP_WORDS)),
                         ('tfidf', TfidfTransformer()),
                         ('nmf', NMF(n_components=50, init='random', random_state=42)),
                         ('clf', SVC(C = 1000, kernel='linear', probability=True)),
])


hard_svc_clf_nmf = hard_svc_clf_nmf.fit(train_data, train_target)
predicted_prob = hard_svc_clf_nmf.predict_proba(test_data)[:, 1]
predicted = hard_svc_clf_nmf.predict(test_data)

fpr, tpr, thresholds = roc_curve(test_target, predicted_prob)
auroc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (auroc = %0.3f)' % auroc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve--Hard Margin SVC using NMF')
plt.legend(loc="lower right")
plt.show()
        
hard_svc_clf_nmf = hard_svc_clf_nmf.fit(train_data, train_target)
predicted_prob = hard_svc_clf_nmf.predict_proba(test_data)[:, 1]
predicted = hard_svc_clf_nmf.predict(test_data)
cm = confusion_matrix(test_target, predicted)
accuracy = accuracy_score(test_target, predicted)
precision = precision_score(test_target, predicted, pos_label = 1)
recall = recall_score(test_target, predicted, pos_label = 1)
print("Result for Hard Margin SVC using NMF")
print("confusion_matrix: ")
print(cm)
print("accuracy = %f, precision = %f, recall＝ %f" %(accuracy, precision, recall))


##--------Soft Margin SVC using LSI----------------
soft_svc_clf_lsi = Pipeline([('vect', CountVectorizer(min_df=5, stop_words=text.ENGLISH_STOP_WORDS)),
                         ('tfidf', TfidfTransformer()),
                         ('svd', TruncatedSVD(n_components=50, n_iter=10,random_state=42)),
                         ('clf', SVC(C = 0.001, kernel='rbf', probability=True)),
])

soft_svc_clf_lsi = soft_svc_clf_lsi.fit(train_data, train_target)
predicted_prob = soft_svc_clf_lsi.predict_proba(test_data)[:, 1]
predicted = soft_svc_clf_lsi.predict(test_data)

fpr, tpr, thresholds = roc_curve(test_target, predicted_prob)
auroc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (auroc = %0.4f)' % auroc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve--Soft Margin SVC using LSI')
plt.legend(loc="lower right")
plt.show()

cm = confusion_matrix(test_target, predicted)
accuracy = accuracy_score(test_target, predicted)
precision = precision_score(test_target, predicted, pos_label = 1)
recall = recall_score(test_target, predicted, pos_label = 1)
print("Result for Soft Margin SVC using LSI")
print("confusion_matrix: ")
print(cm)
print("accuracy = %f, precision = %f, recall＝ %f" %(accuracy, precision, recall))


##--------Soft Margin SVC using NMF----------------
soft_svc_clf_nmf = Pipeline([('vect', CountVectorizer(min_df=5, stop_words=text.ENGLISH_STOP_WORDS)),
                         ('tfidf', TfidfTransformer()),
                         ('nmf', NMF(n_components=50, init='random',random_state=42)),
                         ('clf', SVC(C = 0.001, kernel='rbf', probability=True)),
])
soft_svc_clf_nmf = soft_svc_clf_nmf.fit(train_data, train_target)
predicted_prob = soft_svc_clf_nmf.predict_proba(test_data)[:, 1]
predicted = soft_svc_clf_nmf.predict(test_data)

fpr, tpr, thresholds = roc_curve(test_target, predicted_prob)
auroc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (auroc = %0.4f)' % auroc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Soft Margin SVC using NMF')
plt.legend(loc="lower right")
plt.show()

cm = confusion_matrix(test_target, predicted)
accuracy = accuracy_score(test_target, predicted)
precision = precision_score(test_target, predicted, pos_label = 1)
recall = recall_score(test_target, predicted, pos_label = 1)
print("Result for Soft Margin SVC using NMF")
print("confusion_matrix: ")
print(cm)
print("accuracy = %f, precision = %f, recall＝ %f" %(accuracy, precision, recall))
    
    
 Set the Cross validation
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True)

stop_words = text.ENGLISH_STOP_WORDS
count_vect = CountVectorizer(min_df=2, stop_words=stop_words)
train_counts = count_vect.fit_transform(train_data)
tfidf_transformer = TfidfTransformer()
X = tfidf_transformer.fit_transform(train_counts)

accuracy = []
precision = []
recall = []
gamma = []

file = "svm_five_fold_report/report.txt"
try:
    os.remove(file)
except OSError:
    pass

with open(file, 'a') as f:
    for exp in range(-3, 4):
        cur_gamma = 10**exp
        svm = SVC(C=cur_gamma)
        gamma.append(cur_gamma)

        accuracy_ave = 0.0
        precision_ave = 0.0
        recall_ave = 0.0
        fold = 0
        for train, test in kf.split(X):
            svm = svm.fit(X[train], train_target[train])
            predicted = svm.predict(X[test])
            cm = confusion_matrix(train_target[test], predicted)
            cur_accuracy = accuracy_score(train_target[test], predicted)
            cur_precision = precision_score(train_target[test], predicted, pos_label=1)
            cur_recall = recall_score(train_target[test], predicted, pos_label=1)
            # print("Current gamma = %f, fold = %d" %(cur_gamma, fold))
            f.write("Current gamma = %f, fold = %d\n" %(cur_gamma, fold))
            fold += 1
            # print("confusion matrix = ")
            f.write("confusion matrix = \n")
            # print(cm)
            f.write(str(cm) + "\n")
            # print("accuracy = %f, precision = %f, recall＝ %f" %(cur_accuracy, cur_precision, cur_recall))
            f.write("accuracy = %f, precision = %f, recall = %f\n" %(cur_accuracy, cur_precision, cur_recall))
            accuracy_ave += cur_accuracy
            precision_ave += cur_precision
            recall_ave += cur_recall
        accuracy_ave /= 5
        precision_ave /= 5
        recall_ave /= 5
        accuracy.append(accuracy_ave)
        precision.append(precision_ave)
        recall.append(recall_ave)
        
print("gamma:")
print(gamma)
print("accuracy:")
print(accuracy)
print("precision:")
print(precision)
print("recall:")
print(recall)
# get the best value for accuracy
index = np.argmax(np.array(accuracy))
print("gamma for maximum accuracy: gamma = %f, accuracy = %f" %(gamma[index], accuracy[index]))
index = np.argmax(np.array(precision))
print("gamma for maximum precision: gamma = %f, precision = %f" %(gamma[index], precision[index]))
index = np.argmax(np.array(recall))
print("gamma for maximum recall: gamma = %f, recall = %f" %(gamma[index], recall[index]))

#lsi
soft_svc_clf = Pipeline([('vect', CountVectorizer(min_df=5, stop_words=text.ENGLISH_STOP_WORDS)),
                         ('tfidf', TfidfTransformer()),
                         ('svd', TruncatedSVD(n_components=50, n_iter=10,random_state=42)),
                         ('clf', SVC(C = 10, kernel='rbf', probability=True)),
])
soft_svc_clf = soft_svc_clf.fit(train_data, train_target)
predicted_proba = soft_svc_clf.predict_proba(test_data)[:, 1]
predicted = soft_svc_clf.predict(test_data)

##nmf
soft_svc_clf_nmf = Pipeline([('vect', CountVectorizer(min_df=5, stop_words=text.ENGLISH_STOP_WORDS)),
                         ('tfidf', TfidfTransformer()),
                         ('nmf', NMF(n_components=50, init='random',random_state=42)),
                         ('clf', SVC(C = 1000, kernel='rbf', probability=True)),
])
soft_svc_clf_nmf = soft_svc_clf_nmf.fit(train_data, train_target)
predicted_prob = soft_svc_clf_nmf.predict_proba(test_data)[:, 1]
predicted = soft_svc_clf_nmf.predict(test_data)

 plot roc curve
fpr, tpr, thresholds = roc_curve(test_target, predicted_proba)
auroc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (auroc = %0.5f)' % auroc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve--Soft Margin SVC using NMF (gamma = 1000)')
plt.legend(loc="lower right")
plt.show()

cm = confusion_matrix(test_target, predicted)
accuracy = accuracy_score(test_target, predicted)
precision = precision_score(test_target, predicted, pos_label = 1)
recall = recall_score(test_target, predicted, pos_label = 1)
print("Result for Soft Margin SVC")
print("confusion_matrix: ")
print(cm)
print("accuracy = %f, precision = %f, recall＝ %f" %(accuracy, precision, recall))


##Naive Bayes--lsi
clf = Pipeline([('vect', CountVectorizer(min_df=2, stop_words=text.ENGLISH_STOP_WORDS)),
                ('tfidf', TfidfTransformer()),
                ('svd', TruncatedSVD(n_components=50, n_iter=10, random_state=42)),
                ('clf', GaussianNB()),
])
clf = clf.fit(train_data, train_target)

predicted_prob = clf.predict_proba(test_data)[:, 1]
predicted = clf.predict(test_data)

fpr, tpr, thresholds = roc_curve(test_target, predicted_prob)
auroc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (auroc = %0.3f)' % auroc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve--Naive Bayes Classifier using LSI')
plt.legend(loc="lower right")
plt.show()

cm = confusion_matrix(test_target, predicted)
accuracy = accuracy_score(test_target, predicted)
precision = precision_score(test_target, predicted, pos_label = 1)
recall = recall_score(test_target, predicted, pos_label = 1)
print("Result for Naive Bayes using LSI")
print("confusion_matrix: ")
print(cm)
print("accuracy = %f, precision = %f, recall＝ %f" %(accuracy, precision, recall))


##Naive Bayes--nmf
clf_nmf = Pipeline([('vect', CountVectorizer(min_df=2, stop_words=text.ENGLISH_STOP_WORDS)),
                ('tfidf', TfidfTransformer()),
                ('nmf', NMF(n_components=50, init='random', random_state=42)),
                ('clf', GaussianNB()),
])
clf_nmf = clf_nmf.fit(train_data, train_target)
predicted_prob = clf_nmf.predict_proba(test_data)[:, 1]
predicted = clf_nmf.predict(test_data)
fpr, tpr, thresholds = roc_curve(test_target, predicted_prob)
auroc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (auroc = %0.3f)' % auroc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve--Naive Bayes Classifier using NMF')
plt.legend(loc="lower right")
plt.show()

cm = confusion_matrix(test_target, predicted)
accuracy = accuracy_score(test_target, predicted)
precision = precision_score(test_target, predicted, pos_label = 1)
recall = recall_score(test_target, predicted, pos_label = 1)
print("Result for Naive Bayes Clasifier using NMF")
print("confusion_matrix: ")
print(cm)
print("accuracy = %f, precision = %f, recall＝ %f" %(accuracy, precision, recall))
    
    
###Logistic Regression--lsi
stop_words = text.ENGLISH_STOP_WORDS
count_vect = CountVectorizer(min_df=2, stop_words=stop_words)
train_counts = count_vect.fit_transform(train_data)
test_counts = count_vect.transform(test_data)
tfidf_transformer = TfidfTransformer()
X = tfidf_transformer.fit_transform(train_counts)
y = tfidf_transformer.transform(test_counts)
svd = TruncatedSVD(n_components=50, n_iter=10,random_state=42)
X_svd = svd.fit_transform(X)
y_svd = svd.fit_transform(y)
nmf = NMF(n_components=50, init='random', random_state=42)
X_nmf = nmf.fit_transform(X)
y_nmf = nmf.fit_transform(y)

clf = Pipeline([('vect', CountVectorizer(min_df=2, stop_words=text.ENGLISH_STOP_WORDS)),
                ('tfidf', TfidfTransformer()),
                ('svd', TruncatedSVD(n_components=50, n_iter=10, random_state=42)),
                ('clf', LogisticRegression()),
])
    
clf = LogisticRegression()

clf = clf.fit(X_svd, train_target)
predicted_prob = clf.predict_proba(y_svd)[:, 1]
predicted = clf.predict(y_svd)

fpr, tpr, thresholds = roc_curve(test_target, predicted_prob)
auroc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (auroc = %0.3f)' % auroc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve--Logistic Regression using LSI')
plt.legend(loc="lower right")
plt.show()

cm = confusion_matrix(test_target, predicted)
accuracy = accuracy_score(test_target, predicted)
precision = precision_score(test_target, predicted, pos_label = 1)
recall = recall_score(test_target, predicted, pos_label = 1)
print("Result for Logistic Regression using LSI")
print("confusion_matrix: ")
print(cm)
print("accuracy = %f, precision = %f, recall＝ %f" %(accuracy, precision, recall))


###Logistic Regression--nmf
clf = clf.fit(X_nmf, train_target)
predicted_prob = clf.predict_proba(y_nmf)[:, 1]
predicted = clf.predict(y_nmf)
fpr, tpr, thresholds = roc_curve(test_target, predicted_prob)
auroc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (auroc = %0.3f)' % auroc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve--Logistic Regression using NMF')
plt.legend(loc="lower right")
plt.show()

cm = confusion_matrix(test_target, predicted)
accuracy = accuracy_score(test_target, predicted)
precision = precision_score(test_target, predicted, pos_label = 1)
recall = recall_score(test_target, predicted, pos_label = 1)
print("Result for Logistic Regression using NMF")
print("confusion_matrix: ")
print(cm)
print("accuracy = %f, precision = %f, recall＝ %f" %(accuracy, precision, recall))


### try l1 norm regulations
val_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1, 10, 1e2, 1e3]
file = "logistic_regression_l1/report.txt"
try:
    os.remove(file)
except OSError:
    pass

with open(file, 'a') as f:
    for val in val_list:
        clf = LogisticRegression(penalty="l1", C=val)
        clf.fit(X_svd, train_target)
        predicted = clf.predict(y_svd)
        predicted_proba = clf.predict_proba(y_svd)[:, 1]

        fpr, tpr, thresholds = roc_curve(test_target, predicted_proba)
        auroc = auc(fpr, tpr)
        plt.clf()
        plt.plot(fpr, tpr, label='ROC curve (auroc = %0.3f)' % auroc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve for Logistic Regression, l1 gamma = %f' % val)
        plt.legend(loc="lower right")
        plt.savefig("logistic_regression_l1/roc_curve_l1_C{}.png".format(str(val)))

        cm = confusion_matrix(test_target, predicted)
        accuracy = accuracy_score(test_target, predicted)
        precision = precision_score(test_target, predicted, pos_label = 1)
        recall = recall_score(test_target, predicted, pos_label = 1)
        f.write("Current Gamma = %f\n" % val)
        f.write("confusion matrix = \n")
        f.write(str(cm) + "\n")
        f.write("accuracy = %f, precision = %f, recall = %f\n" %(accuracy, precision, recall))
        
### try l2 norm regulations
val_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1, 10, 1e2, 1e3]
file = "logistic_regression_l2/report.txt"
try:
    os.remove(file)
except OSError:
    pass

with open(file, 'a') as f:
    for val in val_list:
        clf = LogisticRegression(penalty="l2", C=val)
        clf.fit(X, train_target)
        predicted = clf.predict(y)
        predicted_proba = clf.predict_proba(y)[:, 1]

        fpr, tpr, thresholds = roc_curve(test_target, predicted_proba)
        auroc = auc(fpr, tpr)
        plt.clf()
        plt.plot(fpr, tpr, label='ROC curve (auroc = %0.3f)' % auroc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve for Logistic Regression, l2 gamma = %f' % val)
        plt.legend(loc="lower right")
        plt.savefig("logistic_regression_l2/roc_curve_l2_C{}.png".format(str(val)))

        cm = confusion_matrix(test_target, predicted)
        accuracy = accuracy_score(test_target, predicted)
        precision = precision_score(test_target, predicted, pos_label = 1)
        recall = recall_score(test_target, predicted, pos_label = 1)
        f.write("Current Gamma = %f\n" % val)
        f.write("confusion matrix = \n")
        f.write(str(cm) + "\n")
        f.write("accuracy = %f, precision = %f, recall = %f\n" %(accuracy, precision, recall))