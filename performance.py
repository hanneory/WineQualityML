from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import pylab

# Takes the predictions and the lables
def confusion_matrix(P, L):
    CM = np.zeros(2,2)

    for i in range(2):
        for j in range(2):
            CM[i,j] = ((P == i)*(L == j)).sum()

    return CM

# llrs = log likelihood ratios
# does not tell us where to put the thresholds, 
# only the tradeoff between the true positive and the
# false positive rates 
def ROC_plot(llrs, L):
    thresh = np.array(llrs)
    thresh = thresh.sort()
    thresh = np.concatenate([np.array([-np.inf]), thresh, np.array([np.inf])])
    FPR = np.zeros(thresh.size)
    TPR = np.zeros(thresh.size)

    for i, t in enumerate(thresh):
        P = np.int32(llrs > t)
        CM = confusion_matrix(P, L)
        TPR[i] = CM[1,1] / CM[1,1] + CM[0,1]
        FPR[i] = CM[1,0] / CM[1,0] + CM[0,0]

    pylab.plot(FPR, TPR)
    pylab.show()


# S MUST BE LLRS - LOG LIKELIHOOD RATIOS
# WE MIGHT NEED TO TRY DIFFERENT THRESHOLDS

def assign_labels(S, pi, Cfn, Cfp, th = None):
    if th is None:
        th = -np.log(pi * Cfn) + np.log((1-pi) * Cfp)
    P = S > th
    return np.int32(P)

# EMPIRICAL BAYES ERROR FOR BINARY PROBLEMS
def emp_Bayes_bin(CM, pi, Cfn, Cfp):
    fnr = CM[0,1] / CM[0,1] + CM[1,1]
    fpr = CM[1,0] / CM[0,0] + CM[1,0]
    return pi * Cfn * fnr + (1 - pi) * Cfp * fpr

def normalized_emp_Bayes_bin(CM, pi, Cfn, Cfp):
    EB = emp_Bayes_bin(CM, pi, Cfn, Cfp)
    return EB / min(pi*Cfn, (1- pi)*Cfp)

# ACTUAL DETECTION COST FUNCTION
def compute_act_DCF(S, L, pi, Cfn, Cfp, th = None):
    P = assign_labels(S, pi, Cfn, Cfp, th = th)
    CM = confusion_matrix(P, L)
    return normalized_emp_Bayes_bin(CM, pi, Cfn, Cfp)

def compute_min_DCF(S, L, pi, Cfn, Cfp):
    t = np.array(S)
    t = t.sort()
    np.concatenate([np.array([-np.inf]), t, np.array([np.inf])])
    dcfL = []
    for _t in t:
        dcfL.append(compute_act_DCF(S, L, pi, Cfn, Cfp, t))
    return np.array(dcfL).min()

def bayes_error_plot(pA, S, L, minCost=False):
    y = []
    for p in pA:
        pi = 1.0 / (1.0 + np.exp(-p))
        if minCost:
            y.append(compute_min_DCF(S, L, pi, 1, 1))
        else:
            y.append(compute_act_DCF(S, L, pi, 1, 1))
    return np.array(y)

def plot_performance(scores, lables):
    #TODO ADD THE CONFUSION MATRIX AND ROC PLOT, MAYBE DCF PLOT
    #TODO CHECK THAT BAYES COMES OUT RIGHT

    # not work for logistic regression as this does not produce log likelihood
    # different thrsholds if the scores include the priors

    #TODO THIS MIGHT NEED TO CHANGE
    p = np.linspace(-3, 3, 21)
    pylab.plot(p, bayes_error_plot(p, scores, lables, mincost = False), color = "r")
    pylab.plot(p, bayes_error_plot(p, scores, lables, mincost = True), color = "b")

#if log reg model take the score of the reg and transform into new score that behanves  like a llrs
# we remove the proir information to make use of the score. 