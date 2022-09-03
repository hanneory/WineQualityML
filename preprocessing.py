import numpy
import pylab
import matplotlib.pyplot as plt
from sklearn import preprocessing  as pp
import pandas as pd
import scipy.stats
import support_functions as sp


def pca(D, dim):
    DC = D - empirical_mean(D)
    C = numpy.dot(DC, DC.T) / float(D.shape[1])
    _, U = numpy.linalg.eigh(C)
   
    #select larges eigenvalues
    P = U[:, ::-1][:, 0:dim]
    DP = numpy.dot(P.T, D)

    return DP


def gaussianize(DTR, DTE):
    rDTR = numpy.zeros(DTR.shape)
    rDTE = numpy.zeros(DTE.shape)

    # compute rank over training set
    for f in range(DTR.shape[0]):
        for e in range(DTR.shape[1]):
            rDTR[f][e] = (DTR[f] < DTR[f][e]).sum()
    
    rDTR = (rDTR + 1)/(DTR.shape[1] + 2) 

    # rank over test set
    for f in range(DTE.shape[0]):
        for e in range(DTE.shape[1]):
            rDTE[f][e] = (DTE[f] < DTE[f][e]).sum()
    
    rDTE = (rDTE + 1)/(DTR.shape[1] + 2) 

    #compute transformed feature
    yTR = scipy.stats.norm.ppf(rDTR)
    yTE = scipy.stats.norm.ppf(rDTE)

    return yTR, yTE


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def empirical_mean(X):
    return sp.mcol(X.mean())

def zero_values(D):
    df = pd.DataFrame(D.T)
    df.columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

    #replacing the values with another low value 
    condition = (df['citric acid'] == 0)
    df.loc[condition, 'citric acid'] = 0.01

    D = df.to_numpy()

    return D.T
