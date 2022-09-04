import numpy as np
import scipy.optimize as sc
import support_functions as sp
import performance as p
import pylab

# l = lambda
# returns the objective of the function
# J(v)

#TODO might need to rebalance the classes

#The lables does not change from iteration to iteration 
# so they can be computed once in a separate function
# THIS IS F
def logreg_obj_wrap(DTR, LTR, l):
    Z = LTR * 2.0 - 1.0
    M = DTR.shape[0]
    def logreg_obj(v):      # v packs w_b
        w = sp.mcol(v[0:M])
        b = v[-1]
        
        #this is the exponential inside J -> (w.T*xi + b)
        S = np.dot(w.T, DTR) + b
        cxe = np.logaddexp(0, -S*Z).mean()
        #regulizer
        return cxe + 0.5*l*np.linalg.norm(w)**2
    return logreg_obj


def log_reg_classifier(DTR, LTR, DTE, LTE, l):
    #iterate through a predefined set of lambda values
    minDCF = np.zeros([3, len(l)])
    i = 0
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
        print("Error rate: ", sp.accuracy(LTE, LP), "% \n")

        #minDCF5 = p.compute_min_DCF(STE, LTE, 0.5, 1, 1)
        #minDCF9 = p.compute_min_DCF(STE, LTE, 0.9, 1, 1)
        #minDCF1 = p.compute_min_DCF(STE, LTE, 0.1, 1, 1)

        #print(minDCF5)
        #print(minDCF1)
        #print(minDCF9)

        #minDCF[0, i] = minDCF5
        #minDCF[1, i] = minDCF9
        #minDCF[2, i] = minDCF1

        i += 1
    
    p.plot_minDCF("LogReg", "Lambda", l, minDCF[0], minDCF[1], minDCF[2])
        
        

def get_score(_v, LP, pi):
    
    alpha = _v[0]
    betafirst = _v[1]
    scores = alpha*LP + betafirst - np.log(pi / (1 - pi))

    return scores