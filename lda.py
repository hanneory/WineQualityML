import numpy
import scipy.linalg
import pylab
import support_functions as sf

def empirical_covariance(X):
        print(X)
        mu = sf.mcol(X.mean(1))
        print(mu)
        cov = numpy.dot((X-mu),(X-mu).T)/X.shape[1]
        return cov

def within_class_covariance(D, L):
    
    SW = 0
    for i in range(1):
        SW += (L == i).sum() * empirical_covariance(D[:, L == i]) #calculate only for relevant data


    SW = SW / D.shape[1]

    return SW

def between_class_covariance(X, L):

    SB = 0
    # now we use the mean of the dataset not the class
    muTotal = X.mean(1)
    for i in set(list(L)):
        D = X[:, L == i]
        muClass = sf.mcol(D.mean(1))
        SB += D.shape[1] * numpy.dot( (muClass - muTotal), (muClass - muTotal).T )

    # D.shape is num of samples in class
    # X.shape is num of samples in total
    return SB / X.shape[1]

def LDA(D, L):

    SW = within_class_covariance(D, L)
    SB = between_class_covariance(D, L)



    # We want them in opposite directions
    # can find at most two directions since we have three classes

    m = 2
    U = U[:, ::-1][:, 0:m] # Reverse and take the m first columns

    # These are the LDA directions

    print(U)
    #orth = numpy.dot(U, U.T)
    #print(orth) # U not orthogonal, the columns span the subspace but is not a basis

    # if we want a basis we need to orthoganalize it
    # LDA can output a different scale than what we originally had

