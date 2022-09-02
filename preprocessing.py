import numpy
import pylab
import matplotlib.pyplot as plt
from sklearn import preprocessing  as pp
import pandas as pd

def mcol(v):
    return v.reshape((v.size, 1))

def pca(D, dim):
    #mu = D.mean(1)
    #DC = D - mu.reshape((mu.size, 1)) 
    #C = numpy.dot(DC, DC.T) / D.shape[1]
    #s, U = numpy.linalg.eigh(C)
   
    #select larges eigenvalues
    #P = U[:, ::-1][:, 0:dim]

    #NEXT ATTEMPT
    mu = D.mean(1) #as we want to compute over the axis=1 bc columns

    covar_matrix = 0

    # centered dataset
    xc = D - vcol(mu) #mu needs to be a column vector here
    xcxct = numpy.dot(xc, xc.T)  / D.shape[1]
    covar_matrix = covar_matrix + xcxct

    #compute eigenvectors and eigenvalues
    s, U = numpy.linalg.eigh(covar_matrix)

    # s = eigenvalues sorted from smallest to largest 
    # U = corresponding eigenvectors to s

    # We want to extract the m largest eigenvectors from U
    # reverse the order
    # set in columns

    P = U[:, ::-1][:, 0:dim]

    # We can also get them from directly taking a SVD - Singular value decomposition
    U, s, Vh = numpy.linalg.svd(covar_matrix)

    # U and s now sorted in descending order
    # Select the m largest
    P = U[:, 0:dim]

    #project dataset
    DP = numpy.dot(P.T, D)   

    return DP

def gaussianize(D,L):
    bc = pp.PowerTransformer()

    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    D_trans_bc = numpy.zeros([len(D[0]),1])

    for dIdx in range(11):

        X0_trans_bc = bc.fit_transform(D0[dIdx,:].reshape(-1,1), D0[dIdx,:].reshape(-1,1))
        X1_trans_bc = bc.fit_transform(D1[dIdx,:].reshape(-1,1), D1[dIdx,:].reshape(-1,1))

        if dIdx == 7: 
            X1_trans_bc = X1_trans_bc * (3e15)

        D_insert = numpy.append(X0_trans_bc, X1_trans_bc)
        D_insert = numpy.reshape(D_insert, (len(D_insert), 1))
        D_trans_bc = numpy.append(D_trans_bc, D_insert, axis=1)

    D_trans_bc = numpy.delete(D_trans_bc, 0 , axis=1)

    return D_trans_bc.T


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

