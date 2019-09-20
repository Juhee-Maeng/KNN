# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 15:59:23 2018

@author: mjh0208
"""

import knn
import numpy as np
import pandas as pd

train = pd.read_csv("digits_train.csv", header=None)
test = pd.read_csv("digits_test.csv", header=None)

X_train = train.iloc[:,1:]
y_train = train.iloc[:,0]

X_test = test.iloc[:,1:]
y_test = test.iloc[:,0]

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

#최종 predict하는 함수
def kNN_test(X_train, X_test, y_train, y_test, distance = "euclidean",k=3):
    output_classes = []
    for i in range(0, X_test.shape[0]):
        output = knn.train(X_train, X_test[i],distance,k)
        predictedClass = knn.predict(output, y_train)
        output_classes.append(predictedClass)
    return output_classes

#confusion matrix 함수
def confusion_matrix(y_true, y_pred):
    y_true_ = pd.Series(y_true,  name='Actual')
    y_pred_ = pd.Series(y_pred,  name='Predict')
    
    return pd.crosstab(y_true_, y_pred_)

#accuracy 함수
def accuracy(y_true,y_pred):
    c_matrix = np.array(confusion_matrix(y_true, y_pred))
    return np.sum(c_matrix.diagonal())/ len(y_true)

#precision 함수
def precision(y_true, y_pred):
    c_matrix = confusion_matrix(y_true, y_pred) 
    precision_list = []
    for i in range(len(c_matrix)):
        precision_list.append(c_matrix.iloc[i,i]/sum(c_matrix.iloc[:,i]))
    return np.mean(precision_list)

#recall 함수
def recall(y_true, y_pred):
    c_matrix = confusion_matrix(y_true, y_pred) 
    recall_list = []
    for i in range(len(c_matrix)):
        recall_list.append(c_matrix.iloc[i,i]/sum(c_matrix.iloc[i,:]))
    return np.mean(recall_list)

#f1_score 함수
def f1_score(y_true, y_pred):
    c_matrix = confusion_matrix(y_true, y_pred)
    precision_list = []
    recall_list = []
    for i in range(len(c_matrix)):
        precision_list.append(c_matrix.iloc[i,i]/sum(c_matrix.iloc[:,i]))
        recall_list.append(c_matrix.iloc[i,i]/sum(c_matrix.iloc[i,:]))
    
    precision_sum = sum([1/precision for precision in precision_list])
    recall_sum = sum([1/recall for recall in recall_list])
    
    f1 = 2*len(c_matrix) / (precision_sum + recall_sum)
    return f1

# test
y_pred = kNN_test(X_train, X_test, y_train, y_test, distance = "euclidean",k=4)
accuracy(y_test,y_pred)
precision(y_test, y_pred)
recall(y_test, y_pred)
f1_score(y_test, y_pred)




###################################################################
#5-fold cross validation관련
train_data = pd.read_csv("digits_train.csv", header=None)
train_data = np.array(train_data)

train_data.shape[0]
data_size = len(train_data)

#랜덤 배열
random_index = np.random.permutation(train_data.shape[0])

start_index = 0
for i in range(5):
    end_index = start_index + int(data_size/5)
    test_index = random_index[start_index:end_index] #test data 구간 index
    TEST = train_data[test_index] #test data(5fold중 1fold)
    TRAIN = np.delete(train_data,test_index,0)#전체데이터에서 test data 제외한 나머지
    
    X_TEST = TEST[:,1:]
    y_TEST = TEST[:,0]
    X_TRAIN = TRAIN[:,1:]
    y_TRAIN = TRAIN[:,0]
    
    y_pred = kNN_test(X_TRAIN, X_TEST, y_TRAIN, y_TEST, distance = "euclidean",k=3)
    
    print("############################################")
    print("{}_fold:".format(i+1))
    print("accuracy = {}".format(accuracy(y_TEST,y_pred)))
    print("precision = {}".format(precision(y_TEST, y_pred)))
    print("recall = {}".format(recall(y_TEST, y_pred)))
    print("f1_score = {}".format(f1_score(y_TEST, y_pred)))
    start_index = end_index









