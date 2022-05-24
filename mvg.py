#Multivariate Gaussian Classifier

import numpy
import sys
import scipy.special

#row to column-vector
def mcol(v):
    return v.reshape((v.size, 1))

#column to row-vector
def vrow(v):
    return v.reshape((1, v.size))

#Compute covariance
def compute_empirical_cov(X):
    mean = compute_empirical_mean(X)
    cov = numpy.dot( (X-mean), (X-mean).T) / X.shape[1]
    return cov

#Compute mean value
def compute_empirical_mean(X):
    return mcol(X.mean(1))

def logpdf_1sample(x, mu, C):
    P = numpy.linalg.inv(C) #C is numpy array of shape (M,M) representing the covariance matrix, taking the inverse
    res = -0.5 * x.shape[0] * numpy.log(2*numpy.pi)
    res += -0.5 * numpy.linalg.slogdet(C)[1] #slogdet is used to calculate the absolute value
    res += -0.5 * numpy.dot ((x-mu).T, numpy.dot(P, (x-mu))) # 1x1 matrix
    return res.ravel() #transforming res to a one dimensional ploting point that contains only one value

def logpdf_GAU_ND (X, mu, C):
    Y = [logpdf_1sample(X[:, i:i+1], mu, C) for i in range (X.shape[1])] #for each element of X I take the i's element, and cast it as a column vector. Calculate the density and add it to a list
    return numpy.array(Y).ravel() 

def loglikelihood(X, mu, C):
    return logpdf_GAU_ND(X, mu, C).sum() #useful when we optimize. 

def likelihood(X, mu, C):
    Y = numpy.exp(logpdf_GAU_ND(X, mu, C))
    return Y.prod()

#Compute density
def density(X, mu, C):
    return numpy.exp(logpdf_GAU_ND(X, mu, C)) #take the exeptional of the log-density

def computeMVG(DTrain, LTrain, DTest, LTest):
    #estimate the model parameters for each class
    h = {} #place in hashtable where the key is the class (0 or 1).
    for lab in [0,1]:
        mu = compute_empirical_mean(DTrain[:, LTrain == lab])
        C = compute_empirical_cov(DTrain[:, LTrain == lab])
        h[lab] = (mu, C)
    
    #Matrix of joint density
    SJoint = numpy.zeros((2, DTest.shape[1]))

    #Matrix of log joint density
    logSJoint = numpy.zeros((2, DTest.shape[1]))

    #Specificy class priors for each class in list.
    classPriors = [1.0/2.0, 1.0/2.0]

    #Fill the corresponding row with joint and logSjoint
    for lab in [0,1]:
        mu, C = h[lab]
        SJoint[lab, :] = density(DTest, mu, C).ravel() * classPriors[lab]
        logSJoint[lab, :] = logpdf_GAU_ND(DTest, mu, C).ravel() + numpy.log(classPriors[lab])
    
    #Marginal of my densities
    SMarginal = SJoint.sum(0)
    logSMarginal = scipy.special.logsumexp(logSJoint, axis = 0) 
    
    #Compute posterior
    Post1 = SJoint / vrow(SMarginal)
    logPost = logSJoint - vrow(logSMarginal)
    Post2 = numpy.exp(logPost)

    LPred1 = Post1.max(0)
    LPred2 = Post2.max(0)

    accuracy = (LPred1 == LTest).sum()
    accuracy_rate = accuracy/LTest.size

    
    
    
    return(LPred1)
        


    