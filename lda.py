import numpy
import sklearn.datasets
import scipy.linalg

# Goal: implement Linear Discriminant Analysis
def load():
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets

def vcol(V):
    return V.reshape((V.size, 1))


def LDA():
    data = load()
    
    # HVA VAR DENNE TIL????

    # find within and between covariance of each class
    # same tech as when calculating the PCA

    Sw0 = data[:, L == 0]
    Sw1 = data[:, L == 1]
    Sw2 = data[:, L == 2]    

    return 

def empirical_covariance(X):
        
        mu = vcol(X.mean(1))
        xc = X - vcol(mu)
        xcxct = numpy.dot(xc, xc.T)  / X.shape[1]
    
        return xcxct

def within_class_covariance(D, L):
    
    SW = 0
    for i in [0,1,2]:
        SW += (L == i).sum() * empirical_covariance(D[:, L == i]) #calculate only for relevant data

    SW = SW / D.shape(1)
    
    #TODO her er det en feil hvor det sies at vi ikke kan
    #     ta summen av boolske verdier

    return SW

def between_class_covariance(X, L):

    SB = 0
    # now we use the mean of the dataset not the class
    muTotal = vcol(X.mean(1))
    for i in set(list(L)):
        D = X[:, L == i]
        muClass = vcol(D.mean(1))
        SB += D.shape(1) * numpy.dot( (muClass - muTotal), (muClass - muTotal).T ) / X.shape[1]

    # D.shape is num of samples in class
    # X.shape is num of samples in total
    return SB

if __name__ == '__main__':
    D, L = load()

    SW = within_class_covariance(D, L)
    SB = between_class_covariance(D, L)

    s, U = scipy.linalg.eigh(SB, SW)

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


