# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:01:07 2020

@author: dylan
"""
import numpy as np

class LogRegression:
    def __init__(self, learningRate, epsilon):
        self.lr = learningRate
        self.eps = epsilon

    def _logistic(self, z):
        yh = 1/(1+np.exp(-z))
        return yh

    def _gradient(self, X, y, w):
        yh = self._logistic(np.dot(X, w))
        gradJ = np.dot(X.T, yh - y)
        return gradJ

    def fit (self, X, y):
        N,D = X.shape
        w = np.zeros(D)
        g = np.inf
        while np.linalg.norm(g) > self.eps:
            g = self._gradient(X, y, w)
            w = w - self.lr*(g)
            return w

    def predict(self,X,w):
        yh = self._logistic(np.dot(X, w))
        yh = np.rint(yh) 
        # This (above) converts each value in yh to the nearest integer (0 or 1)
        return yh
    




