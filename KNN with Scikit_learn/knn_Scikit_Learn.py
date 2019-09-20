# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 16:26:15 2018

@author: mjh0208
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

Train_data = pd.read_csv('cancer_train.csv', header = None) # col의 이름이 없는 경우 header=none
Test_data = pd.read_csv('cancer_test.csv', header = None)
# Data = Train_data.values #array로 변환


y = Train_data.iloc[:,0]
X = Train_data.iloc[:,1:]

y_test = Test_data.iloc[:,0]
X_test = Test_data.iloc[:,1:]

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X,y) 

#train의 predict값을 구하고 accuracy 측정
train_predict = neigh.predict(X)
accuracy_score(y,train_predict) #97.4%의 정확성을 보임

#test의 predit값을 구하고 accuracy 측정
test_predict = neigh.predict(X_test)
accuracy_score(y_test,test_predict) #96.4%의 정확성을 보임

#np.savetxt("foo.csv", train_predict, delimiter=",")
