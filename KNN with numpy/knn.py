# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 23:55:06 2018

@author: mjh0208
"""
import pandas as pd
import numpy as np
import operator

def train(X_train,X_test,distance,K):
    
    distances = []
    neighbors = []
    
    data_num = len(X_train)
    for i in range(0,data_num):
        dist = get_distance(X_train[i],X_test,distance)
        distances.append((i,dist))
    distances.sort(key=operator.itemgetter(1))
    
    for x in range(K):
        neighbors.append(distances[x][0])
    return neighbors



def predict(neighbors, y_train):
    classVotes = {}
    for i in range(len(neighbors)):
        if(y_train[neighbors[i]] in classVotes):
            classVotes[y_train[neighbors[i]]] += 1
        else:
            classVotes[y_train[neighbors[i]]] = 1
    sortedVotes = sorted(classVotes.items(), key = operator.itemgetter(1),reverse = True)
    return sortedVotes[0][0]




def get_distance(x,y,distance):
    if(distance=="euclidean"):
        dist = np.linalg.norm(x-y)
    elif(distance=="manhattan"):
        dist = sum(abs(x-y))
    elif(distance=="maximum_norm"):
        dist = max(abs(x-y))
    return dist
