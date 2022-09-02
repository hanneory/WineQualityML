from distutils.debug import DEBUG
import scipy
import numpy as np

def vcol(V):
    return V.reshape((V.size, 1))

def vrow(V):
    return V.reshape((1, V.size))


# LAB SPESIFIC FUNCTIONS
def empirical_mean(X):
    return vcol(X.mean(1))

def empirical_covariance(X):
        
        mu = vcol(X.mean(1))
        xc = X - vcol(mu)
        xcxct = np.dot(xc, xc.T)  / X.shape[1]
    
        return xcxct


def logpdf_GAU_ND(X, mu, C):
    P = np.linalg.inv(C)
    return -0.5*X.shape[0]*np.log(np.pi*2) + 0.5*np.linalg.slogdet(P)[1] - 0.5*(np.dot(P, (X-mu))*(X-mu)).sum(0)

def ML_GAU(D):
    mu = vcol(D.mean(1))
    C = np.dot(D-mu, (D-mu).T)/float(D.shape[1])
    return mu, C

def within_class_covariance(D, L):
    
    SW = 0
    for i in [0,1]:
        SW += (L == i).sum() * empirical_covariance(D[:, L == i]) #calculate only for relevant data

    SW = SW / D.shape[1]

    return SW


#CLASSIFIERS
def GAU(priors, D_train, L_train, D_test, L_test):
    h = {}

    for i in [0, 1]:
        DX = D_train[:, L_train == i]
        mu, C = ML_GAU(DX)
        h[i] = (mu, C)

    #we now have one table for each class

    SJoint = np.zeros((2, D_test.shape[1]))
    logSJoint = np.zeros((2, D_test.shape[1]))

    classPriors = priors

    for i in [0,1]:
        #compute class conditional densities
        mu, C = h[i]
        SJoint[i, :] = np.exp(logpdf_GAU_ND(D_test, mu, C).ravel()) * classPriors[i]
        logSJoint[i, :] = logpdf_GAU_ND(D_test, mu, C).ravel() * np.log(classPriors[i])

    SMarginal = SJoint.sum(0)
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)

    Post1 = SJoint / vrow(SMarginal)
    #not log
    logPost = logSJoint - vrow(logSMarginal)
    Post2 = np.exp(logPost)
    #log

    #print this
    LPred1 = Post1.argmax(axis=0)
    accuracy = (L_test == LPred1).sum() / L_test.size
    errors = 1 - accuracy

    print("MULTIVARIANT GAUSSIAN CLASSIFIER")
    print("Accuracy: ", accuracy*100, "%")
    print("")


def GAU_DIAG(priors, D_train, L_train, D_test, L_test):
    h = {}

    for i in [0, 1]:
        DX = D_train[:, L_train == i]
        mu, C = ML_GAU(DX)
        C = np.diag(np.diag(C))
        h[i] = (mu, C)

    #we now have one table for each class

    SJoint = np.zeros((2, D_test.shape[1]))
    logSJoint = np.zeros((2, D_test.shape[1]))

    classPriors = priors

    for i in [0,1]:
        #compute class conditional densities
        mu, C = h[i]
        SJoint[i, :] = np.exp(logpdf_GAU_ND(D_test, mu, C).ravel()) * classPriors[i]
        logSJoint[i, :] = logpdf_GAU_ND(D_test, mu, C).ravel() * np.log(classPriors[i])

    SMarginal = SJoint.sum(0)
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)

    Post1 = SJoint / vrow(SMarginal)
    #not log
    logPost = logSJoint - vrow(logSMarginal)
    Post2 = np.exp(logPost)
    #log

    #print this
    LPred1 = Post1.argmax(axis=0)
    accuracy = (L_test == LPred1).sum() / L_test.size
    errors = 1 - accuracy

    print("NAIVE BAYES GAUSSIAN CLASSIFIER")
    print("Accuracy: ", accuracy*100, "%")
    print("")


def GAU_TC(priors, D_train, L_train, D_test, L_test):
    h = {}

    for i in [0, 1]:
        DX = D_train[:, L_train == i]
        mu, C = ML_GAU(DX)

        C = within_class_covariance(D_train, L_train)

        h[i] = (mu, C)

    #we now have one table for each class

    SJoint = np.zeros((2, D_test.shape[1]))
    logSJoint = np.zeros((2, D_test.shape[1]))

    classPriors = priors

    for i in [0,1]:
        #compute class conditional densities
        mu, C = h[i]
        SJoint[i, :] = np.exp(logpdf_GAU_ND(D_test, mu, C).ravel()) * classPriors[i]
        logSJoint[i, :] = logpdf_GAU_ND(D_test, mu, C).ravel() * np.log(classPriors[i])

    SMarginal = SJoint.sum(0)
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)

    Post1 = SJoint / vrow(SMarginal)
    #not log
    logPost = logSJoint - vrow(logSMarginal)
    Post2 = np.exp(logPost)
    #log

    #print this
    LPred1 = Post1.argmax(axis=0)
    accuracy = (L_test == LPred1).sum() / L_test.size
    errors = 1 - accuracy

    print("TIED COVARIANCE GAUSSIAN CLASSIFIER")
    print("Accuracy: ", accuracy*100, "%")
    print("")


def GAU_TC_DIAG(priors, D_train, L_train, D_test, L_test):
    h = {}

    for i in [0, 1]:
        DX = D_train[:, L_train == i]
        mu, _ = ML_GAU(DX)

        C = within_class_covariance(D_train, L_train)
        C = np.diag(np.diag(C))

        h[i] = (mu, C)

    #we now have one table for each class

    SJoint = np.zeros((2, D_test.shape[1]))
    logSJoint = np.zeros((2, D_test.shape[1]))

    classPriors = priors

    for i in [0,1]:
        #compute class conditional densities
        mu, C = h[i]
        SJoint[i, :] = np.exp(logpdf_GAU_ND(D_test, mu, C).ravel()) * classPriors[i]
        logSJoint[i, :] = logpdf_GAU_ND(D_test, mu, C).ravel() * np.log(classPriors[i])

    SMarginal = SJoint.sum(0)
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)

    Post1 = SJoint / vrow(SMarginal)
    #not log
    logPost = logSJoint - vrow(logSMarginal)
    Post2 = np.exp(logPost)
    #log

    #print this
    LPred1 = Post1.argmax(axis=0)
    accuracy = (L_test == LPred1).sum() / L_test.size
    errors = 1 - accuracy

    print("TIED COVARIANCE DIAGNOAL GAUSSIAN CLASSIFIER")
    print("Accuracy: ", accuracy*100, "%")
    print("")

   
def gaussian_classifiers(priors, D_train, L_train, D_test, L_test):
    GAU(priors, D_train, L_train, D_test, L_test)
    GAU_DIAG(priors, D_train, L_train, D_test, L_test)
    GAU_TC(priors, D_train, L_train, D_test, L_test)
    GAU_TC_DIAG(priors, D_train, L_train, D_test, L_test)
