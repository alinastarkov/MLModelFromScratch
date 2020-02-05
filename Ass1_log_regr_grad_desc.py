# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:01:07 2020

@author: dylan
"""

import numpy as np

def logistic(z):
    yh = 1/(1+np.exp(-z))
    return yh

def gradient(X, y, w):
    yh = logistic(np.dot(X, w))
    gradJ = np.dot(X.T, yh - y)
    return gradJ

def fit(X, # N x D        # This is gradient descent 
        y, # N
        learningRate, # learning rate
        eps, # termination codition
        ):
    N,D = X.shape
    w = np.zeros(D)
    g = np.inf
    while np.linalg.norm(g) > eps:
        g = gradient(X, y, w)
        w = w - learningRate*g
        return w

def predict(X,w):
    yh = logistic(np.dot(X, w))
    yh = np.rint(yh) 
    # This (above) converts each value in yh to the nearest integer (0 or 1)
    return yh




