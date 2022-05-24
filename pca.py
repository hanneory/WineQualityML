import numpy
import pylab
import matplotlib.pyplot as plt

def pca(Dmatrix):
    mu = Dmatrix.mean(1)
    DC = Dmatrix - mu.reshape((mu.size, 1)) 
    C = numpy.dot(DC, DC.T) / Dmatrix.shape[1]
    print(C)
    s, U = numpy.linalg.eigh(C)
    m = 2 
    P = U[:, ::-1][:, 0:m]
    DP = numpy.dot(P.T, Dmatrix)
    pylab.scatter(DP[0], DP[1])
    plt.show()