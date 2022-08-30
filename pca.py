import numpy
import pylab
import matplotlib.pyplot as plt

def pca(Dmatrix):
    mu = Dmatrix.mean(1)
    DC = Dmatrix - mu.reshape((mu.size, 1)) 
    C = numpy.dot(DC, DC.T) / Dmatrix.shape[1]
    s, U = numpy.linalg.eigh(C)
    m = 8 
    P = U[:, ::-1][:, 0:m]
    print(P)