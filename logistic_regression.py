import numpy as np
import scipy.optimize as sc
import sklearn.datasets

#change to column-vector
def mcol(v):
    return v.reshape((v.size, 1))

# l = lambda
# returns the objective of the function
# J(v)

#The lables does not change from iteration to iteration 
# so they can be computed once in a separate function
# THIS IS F
def logreg_obj_wrap(DTR, LTR, l):
    Z = LTR * 2.0 - 1.0
    M = DTR.shape[0]
    def logreg_obj(v):      # v packs w_b
        w = mcol(v[0:M])
        b = v[-1]
        
        #this is the exponential inside J -> (w.T*xi + b)
        S = np.dot(w.T, DTR) + b
        cxe = np.logaddexp(0, -S*Z).mean()
        #regulizer
        return cxe + 0.5*l*np.linalg.norm(w)**2
    return logreg_obj


def accuracy(LTE, Lpred):
    return (1 - (np.sum(LTE==Lpred) / len(LTE)))*100


def log_reg_classifier(DTR, LTR, DTE, LTE, l):
    #iterate through a predefined set of lambda values
    for lamb in l:
        logreg_obj = logreg_obj_wrap(DTR, LTR, lamb)
        _v, _J, _d = sc.fmin_l_bfgs_b(logreg_obj, np.zeros(DTR.shape[1]), approx_grad=True)
        _w = _v[0:DTR.shape[0]]
        _b = _v[-1]
        STE = np.dot(_w.T, DTE) + _b
        LP = np.array([])
        for elem in range(STE.size):
            if STE[elem] > 0:
                LP = np.append(LP, True)
            else:
                LP = np.append(LP, False)
        
        print("Logistic Regression with lambda =", lamb)
        print("Error rate: ", accuracy(LTE, LP), "% \n")