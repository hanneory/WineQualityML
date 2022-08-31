import numpy
import pylab
import matplotlib.pyplot as plt
from sklearn import preprocessing  as pp

def mcol(v):
    return v.reshape((v.size, 1))

def pca(Dmatrix):
    mu = Dmatrix.mean(1)
    DC = Dmatrix - mu.reshape((mu.size, 1)) 
    C = numpy.dot(DC, DC.T) / Dmatrix.shape[1]
    s, U = numpy.linalg.eigh(C)
    m = 8 
    P = U[:, ::-1][:, 0:m]

def gaussianize(D,L):
    bc = pp.PowerTransformer()

    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    D_trans_bc = numpy.zeros([1839,1])

    for dIdx in range(11):

        X0_trans_bc = bc.fit_transform(D0[dIdx,:].reshape(-1,1), D0[dIdx,:].reshape(-1,1))
        X1_trans_bc = bc.fit_transform(D1[dIdx,:].reshape(-1,1), D1[dIdx,:].reshape(-1,1))

        if dIdx == 7: 
            X1_trans_bc = X1_trans_bc * (3e15)

        D_insert = numpy.append(X0_trans_bc, X1_trans_bc)
        D_trans_bc = numpy.append(D_trans_bc, mcol(D_insert), axis=1)

    D_trans_bc = numpy.delete(D_trans_bc, 0 , axis=1)

    return D_trans_bc