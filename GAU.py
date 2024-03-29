from distutils.debug import DEBUG
import scipy
import numpy as np
import support_functions as sp
import performance as pf

# LAB SPESIFIC FUNCTIONS
def empirical_mean(X):
    return sp.mcol(X.mean(1))

def empirical_covariance(X):
        
        mu = sp.mcol(X.mean(1))
        xc = X - sp.mcol(mu)
        xcxct = np.dot(xc, xc.T)  / X.shape[1]
    
        return xcxct

def logpdf_GAU_ND(X, mu, C):
    P = np.linalg.inv(C)
    return -0.5*X.shape[0]*np.log(np.pi*2) + 0.5*np.linalg.slogdet(P)[1] - 0.5*(np.dot(P, (X-mu))*(X-mu)).sum(0)

def ML_GAU(D):
    mu = sp.mcol(D.mean(1))
    C = np.dot(D-mu, (D-mu).T)/float(D.shape[1])
    return mu, C

def within_class_covariance(D, L):
    
    SW = 0
    for i in [0,1]:
        SW += (L == i).sum() * empirical_covariance(D[:, L == i]) #calculate only for relevant data

    SW = SW / D.shape[1]

    return SW


def GAU(modelName, D_train, L_train, D_test, L_test):
    h = {}

    if modelName == "DIAG_TC":
        for i in [0, 1]:
            DX = D_train[:, L_train == i]
            mu, _ = ML_GAU(DX)

            C = within_class_covariance(D_train, L_train)
            C = np.diag(np.diag(C))

            h[i] = (mu, C)
    
    elif modelName == "DIAG":
        for i in [0, 1]:
            DX = D_train[:, L_train == i]
            mu, C = ML_GAU(DX)
            C = np.diag(np.diag(C))
            h[i] = (mu, C)
    
    elif modelName == "TC":
        for i in [0, 1]:
            DX = D_train[:, L_train == i]
            mu, C = ML_GAU(DX)

            C = within_class_covariance(D_train, L_train)

            h[i] = (mu, C)

    else:
        for i in [0, 1]:
            DX = D_train[:, L_train == i]
            mu, C = ML_GAU(DX)
            h[i] = (mu, C)

    #we now have one table for each class

    SJoint = np.zeros((2, D_test.shape[1]))
    logSJoint = np.zeros((2, D_test.shape[1]))

    classPriors = [[0.5, 0.5], [0.3, 0.7], [0.7, 0.3]]

    LPred = {}
    logPost= {}
    j = 0

    for p in classPriors:
        for i in [0,1]:
            #compute class conditional densities
            mu, C = h[i]
            SJoint[i, :] = np.exp(logpdf_GAU_ND(D_test, mu, C).ravel()) * classPriors[j][i]
            logSJoint[i, :] = logpdf_GAU_ND(D_test, mu, C).ravel() * np.log(classPriors[j][i])

        SMarginal = SJoint.sum(0)
        logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)

        Post1 = SJoint / sp.mrow(SMarginal)
        logPost[j] = logSJoint - sp.mrow(logSMarginal)
        
        LPred[j] = Post1.argmax(axis=0)
        j += 1

    return LPred[0], LPred[1], LPred[2]

    

   
def gaussian_classifiers(type, D_train, L_train, D_test, L_test):
    if type == "FULL":
        LP_GAU5, LP_GAU1, LP_GAU9  = GAU("MV",  D_train, L_train, D_test, L_test)
    if type == "DIAG":
        LP_GAU5, LP_GAU1, LP_GAU9 = GAU("DIAG",  D_train, L_train, D_test, L_test)
    if type == "TC":
        LP_GAU5, LP_GAU1, LP_GAU9 = GAU("TC",  D_train, L_train, D_test, L_test)
    if type == "DIAG_TC":
        LP_GAU5, LP_GAU1, LP_GAU9 = GAU("DIAG_TC",  D_train, L_train, D_test, L_test)

    return LP_GAU5, LP_GAU1, LP_GAU9



