import numpy
import pylab
import matplotlib.pyplot as plt
from sklearn import preprocessing  as pp
import pandas as pd
import scipy.stats

def mcol(v):
    return v.reshape((v.size, 1))

def pca0(D, dim):
    mu = D.mean(1)
    DC = D - mu.reshape((mu.size, 1)) 
    C = numpy.dot(DC, DC.T) / D.shape[1]
    s, U = numpy.linalg.eigh(C)
   
    #select larges eigenvalues
    P = U[:, ::-1][:, 0:dim]
    DP = numpy.dot(P.T, D)

    return DP

def pca(D, dim):
    #NEXT ATTEMPT
    mu = D.mean(1) #as we want to compute over the axis=1 bc columns

    # centered dataset
    xc = D - vcol(mu) #mu needs to be a column vector here
    C = numpy.dot(xc, xc.T)  / D.shape[1]

    # We can also get them from directly taking a SVD - Singular value decomposition
    U, s, Vh = numpy.linalg.svd(C)

    # U and s now sorted in descending order
    P = U[:, 0:dim]

    #project dataset
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

def vcol(V):
    return V.reshape((V.size, 1))

def empirical_mean(X):
    return vcol(X.mean())

def zero_values(D):
    df = pd.DataFrame(D.T)
    df.columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    #print((df == 0).sum())

    #removing all rows with 0 - but this result in an unmatch between L and D
    #df = df.loc[(df!=0).all(axis=1)]

    #replacing the values with another low value - lowest in the row?
    condition = (df['citric acid'] == 0)
    df.loc[condition, 'citric acid'] = 0.01

    #print("AFTER HANDLING")
    #print((df == 0).sum())

    D = df.to_numpy()

    #print("DTR AFTER HANDLING")
    #print(D.T[2])


    return D.T

