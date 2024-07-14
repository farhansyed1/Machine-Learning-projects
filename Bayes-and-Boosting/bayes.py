#!/usr/bin/python
# coding: utf-8

# Bayes Classifier and Boosting

import numpy as np
from scipy import misc
from importlib import reload
from labfuns import *
import math
import random

def computePrior(labels, W):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))
        
    for clas in classes: 
       idx = np.where(labels == clas)
       prior[clas] = np.sum(W[idx]) / np.sum(W)
  
    return prior

occurrencesInClass = []
# in:      X - N x d matrix of N data points
#     labels - N vector of class labels
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
def mlParams(X, labels, W):
    assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)
    occurrencesInClass = np.zeros(Nclasses)
    Nkw = np.zeros(Nclasses)	

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))

    for clas in classes: 
        sum_X=0
        idx = np.where(labels == clas)
        X_class = X[idx]
        W_class=W[idx]
        
        mu[clas]=np.sum(X_class*W_class, axis=0)/np.sum(W_class)

    for clas in classes:
        idx = np.where(labels == clas)
        X_class = X[idx]
        W_class=W[idx]
    
        # Compute variance for each feature dimension
        for j in range(Ndims):
            square = (X_class[:, j] - mu[clas][j])**2
            sigma[clas][j][j] = np.dot(square,W_class)/np.sum(W_class)
            
    return mu, sigma

# in:      X - N x d matrix of M data points
#      prior - C x 1 matrix of class priors
#         mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
# out:     h - N vector of class predictions for test points
def classifyBayes(X, prior, mu, sigma):

    Npts = X.shape[0]
    Nclasses,Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))

    for clas in range(0,Nclasses):
        term1 = -0.5 * np.log(np.linalg.det(sigma[clas])) 
        # term2 = -0.5*(X-mu[clas])*
        term3 = np.log(prior[clas])
        
        sigma_inv = np.linalg.inv(sigma[clas])

        # term2 = -0.5*np.dot(np.dot((X[data, :]-mu[class_iter]), np.linalg.inv(
        #         sigma[class_iter])), (X[data, :]-mu[class_iter]).transpose())
                
       # logProb = term1 + term2 + term3
        
        for i in range(0,Npts):
            diff = X[i] - mu[clas]
            term2 = -0.5 * np.dot(diff.T, np.dot(sigma_inv, diff))
            logProb[clas, i] = term1 + term2 + term3
    

    h = np.argmax(logProb,axis=0)
    return h

class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)


X, labels = genBlobs(centers=5)

W = np.ones((len(labels), 1)) / len(X)



mu, sigma = mlParams(X,labels,W)

computePrior(labels, W)

# testClassifier(BayesClassifier(), dataset='iris', split=0.7)
# testClassifier(BayesClassifier(), dataset='vowel', split=0.7)
# plotBoundary(BayesClassifier(), dataset='vowel',split=0.7)


# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#                   T - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
#              alphas - (maximum) length T Python list of vote weights
def trainBoost(base_classifier, X, labels, T=10):

    Npts,Ndims = np.shape(X)

    classifiers = [] # append new classifiers to this list
    alphas = [] # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts,1))/float(Npts)

    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))

        # do classification for each point
        vote = classifiers[-1].classify(X)

        delta = np.reshape((vote == labels), (Npts,1))

        err = np.sum(wCur*(1-delta)) +1e-20
    
        alpha = 0.5*(np.log(1-err)-np.log(err))

        for i in range(len(X)):
            if delta[i] == 1:
                wCur[i] = wCur[i]*math.exp(-alpha)
            else:
                wCur[i] = wCur[i]*math.exp(alpha)
        
      
        wCur /= np.sum(wCur)

        alphas.append(alpha)

        
    return classifiers, alphas

# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    Nclasses - the number of different classes
# out:  yPred - N vector of class predictions for test points
def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts,Nclasses))
        
        for t in range(0,Ncomps):
            classified = classifiers[t].classify(X)
            for i in range(0,Npts):
                votes[i][classified[i]] += alphas[t]

        return np.argmax(votes,axis=1)

class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)


# Run some experiments
#testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='iris',split=0.7)

# testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='vowel',split=0.7)

# plotBoundary(BoostClassifier(BayesClassifier()), dataset='vowel',split=0.7)

#testClassifier(DecisionTreeClassifier(), dataset='iris', split=0.7)
# testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)
#testClassifier(DecisionTreeClassifier(), dataset='vowel',split=0.7)

testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel',split=0.7)
#plotBoundary(DecisionTreeClassifier(), dataset='vowel',split=0.7)

plotBoundary(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel',split=0.7)
