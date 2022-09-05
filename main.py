#Use snake_case for the names (is done in the exercises)
#all comments in english. At least one descriptive comments for each function.
#Use multiple files. 

#1 - fixed acidity
#2 - volatile acidity
#3 - citric acid
#4 - residual sugar
#5 - chlorides
#6 - free sulfur dioxide
#7 - total sulfur dioxide
#8 - density
#9 - pH
#10 - sulphates
#11 - alcohol
#Output variable (based on sensory data):
#12 - quality (score between 0 and 10)

#https://towardsdatascience.com/predicting-wine-quality-with-several-classification-techniques-179038ea6434

from ctypes.wintypes import LPRECT
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb



import LR as l
import SVM as s
import GAU as g
import support_functions as sf
import preprocessing as pp
import performance as pf



#load file and and changes row and column. Can't get sklearn to work on mac. 
def load(fname):
    DList = []
    labelList = []

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:11]
                attrs = sf.mcol(numpy.array([float(i) for i in attrs])) 
                name = line.split(',')[-1].strip()
                DList.append(attrs)
                labelList.append(float(name))
            except: #har except fordi det kan hende f.eks siste linja i fila ikkje er lik resten så funksjonane over fungerer ikkje.
                pass
    return numpy.hstack(DList), numpy.array(labelList, dtype = numpy.int32) #Liste med vectors som eg stacke horisontalt som vil gi meg ei matrise. Har også list of labels som lager label-array.

def shuffle(D, L):
    numpy.random.seed(0)
    i = numpy.random.permutation(D.shape[1])
    return D[:, i], L[i]


def kFold(D, L, nFolds, modelName, version = None):
    S = []
    D_sets = numpy.array_split(D, nFolds, axis=1)
    L_sets = numpy.array_split(L, nFolds)
    LPred = []
    L = []

    #for MVG
    LP_GAU5 = []
    LP_GAU3 = []
    LP_GAU7 = []
    lbls = ["raw", "raw gauss"]
    dims = [10, 9, 8]
    for i in range(len(dims)):
        lbls.append("dim %d" % dims[i])
    
    #for LR
    lamb = [1e-6, 1e-3, 0.1]

    #for SVM
    C = [0.01, 0.1, 1]
    K = 1
    gamma = [-2, -1]
    c = [1, 2]
    d = [1, 2]

    for i in range(nFolds):
        DTR, LTR = numpy.hstack(D_sets[:i] + D_sets[i+1:]), numpy.hstack(L_sets[:i] + L_sets[i+1:])
        DTE, LTE = numpy.asanyarray(D_sets[i]), numpy.asanyarray(L_sets[i])

        #run model
        if modelName == "MVG":
            i = 0

            LP5, LP3, LP7 = g.gaussian_classifiers(version, DTR, LTR, DTE, LTE)
            LP_GAU5.append(LP5)
            LP_GAU3.append(LP3)
            LP_GAU7.append(LP7)
            L.append(LTE)
            i += 1
            
            DTR_g, DTE_g = pp.gaussianize(DTR, DTE)
            LP5, LP3, LP7 = g.gaussian_classifiers(version, DTR_g, LTR, DTE_g, LTE)
            LP_GAU5.append(LP5)
            LP_GAU3.append(LP3)
            LP_GAU7.append(LP7)
            L.append(LTE)
            i += 1
            
            for _dim in dims:
                DTR_pg = pp.pca(DTR_g, _dim)
                DTE_pg = pp.pca(DTE_g, _dim)

                LP5, LP3, LP7 = g.gaussian_classifiers(version, DTR_pg, LTR, DTE_pg, LTE)
                LP_GAU5.append(LP5)
                LP_GAU3.append(LP3)
                LP_GAU7.append(LP7)
                L.append(LTE)

        elif modelName == "LR":
            print("Running fold", i)
            if version == ("RAW" or None):
                LPred.append(l.log_reg_classifier(DTR, LTR, DTE, LTE, lamb))
                L.append(LTE)
            
            if version == "GAUS":
                DTR_g, DTE_g = pp.gaussianize(DTR, DTE)
                LPred.append(l.log_reg_classifier(DTR_g, LTR, DTE_g, LTE, lamb))
                L.append(LTE)
        
        elif modelName == "SVM":

            print("Running fold", i)
            
            if version == ("LIN" or None):
                LP = s.SVM_linear_classifier(DTR, LTR, DTE, C, K)
                LPred.append(LP)
                L.append(LTE)
            
            if version == ("POLY"):
                s.SVM_poly_classifier(DTR, LTR, DTE, LTE, C, K, c, d)

            if version == ("RBF"):
                s.SVM_RBF_classifier(DTR, LTR, DTE, LTE, C, K, c, gamma)

            
        elif modelName == "GMM":

            continue

    
    if modelName == "MVG":
        print("minDCF for:" , version)
        for i in range((len(dims)+2)):     
            minDCF5 = pf.compute_min_DCF(numpy.hstack(LP_GAU5[i::(len(dims)+2)]), numpy.hstack(L[i::(len(dims)+2)]), 0.5, 1, 1)
            minDCF3 = pf.compute_min_DCF(numpy.hstack(LP_GAU3[i::(len(dims)+2)]), numpy.hstack(L[i::(len(dims)+2)]), 0.5, 1, 1)
            minDCF7 = pf.compute_min_DCF(numpy.hstack(LP_GAU7[i::(len(dims)+2)]), numpy.hstack(L[i::(len(dims)+2)]), 0.5, 1, 1)
            # unbalanced priors has been taken into account at classification
            print("minDCF for",lbls[i], "minDCF5: ", round(minDCF5,3), "minDCF3", round(minDCF3, 3), "minDCF7", round(minDCF7,3))

    if modelName == "LR":
        lst = []
        minDCF5A = []
        minDCF3A = []
        minDCF7A = []
        for i in range(len(lamb)):
            for j in range(len(LPred)):
                lst.append(LPred[j][i])

            minDCF5 = pf.compute_min_DCF(numpy.hstack(lst), numpy.hstack(L), 0.5, 1, 1)
            minDCF3 = pf.compute_min_DCF(numpy.hstack(lst), numpy.hstack(L), 0.3, 1, 1)
            minDCF7 = pf.compute_min_DCF(numpy.hstack(lst), numpy.hstack(L), 0.7, 1, 1)
            print("                       pi:  0.5   0.3   0.7")
            print("minDCF for lamb =", lamb[i],":", round(minDCF5, 3)," ", round(minDCF3,3)," ", round(minDCF7, 3))

            minDCF3A.append(minDCF3)
            minDCF5A.append(minDCF5)
            minDCF7A.append(minDCF7)
            lst = []
        pf.plot_minDCF("SVM Linear", "lambda", lamb, minDCF5A, minDCF3A, minDCF7A)


    if modelName == "SVM":
        lst = []
        minDCF3A = []
        minDCF5A = []
        minDCF7A = []
        if version == ("LIN" or None):
            for i in range(len(C)):
                for j in range(len(LPred)):
                    lst.append(LPred[j][i])

                minDCF5 = pf.compute_min_DCF(numpy.hstack(lst), numpy.hstack(L), 0.5, 1, 1)
                minDCF3 = pf.compute_min_DCF(numpy.hstack(lst), numpy.hstack(L), 0.3, 1, 1)
                minDCF7 = pf.compute_min_DCF(numpy.hstack(lst), numpy.hstack(L), 0.7, 1, 1)
                minDCF3A.append(minDCF3)
                minDCF5A.append(minDCF5)
                minDCF7A.append(minDCF7)
                print("minDCF for C =", C[i])
                print("minDCF5: ", minDCF5, "minDCF1: ", minDCF3, "minDCF9", minDCF7)
                lst = []
                
            pf.plot_minDCF("SVM Linear", "C", C, minDCF5A, minDCF3A, minDCF7A)
            


if __name__ == '__main__':

    #--------------------------------------------------LOAD DATA----------------------------------------------------------------
   
    D_train, L_train = load('Data/Train.txt')
    D_test, L_test = load('Data/Test.txt')


    #------------------------------------------------VISUALIZATION--------------------------------------------------------------
    
    #sf.plot_scatter(D_train, L_train)
    #sf.plot_gaus(DTR, LTR, DTE, LTE)
    #sf.plot_general_data()


    #---------------------------------------------Logistic regression-----------------------------------------------------------
    
    #print("LOGISITC REGRESSION CLASSIFICATION")
    #print("*************************************")

    print("RAW DATA")
    kFold(D_train, L_train, 5, "LR", "RAW")

    print("GAUSSIANIZED DATA")
    kFold(D_train, L_train, 5, "LR", "GAUS")

    #----------------------------------------------------MVG-----------------------------------------------------------
    
    #print("MULTIVARIANT GAUSSIAN CLASSIFICATION")
    #print("*************************************")

    #kFold(D_train, L_train, 5, "MVG", "FULL")
    #kFold(D_train, L_train, 5, "MVG", "DIAG")
    #kFold(D_train, L_train, 5, "MVG", "TC")
    #kFold(D_train, L_train, 5, "MVG", "DIAG_TC")


    #---------------------------------------Mixed Model Gaussian Classifier-----------------------------------------------------

    #print("MIXED MODEL GAUSSIAN CLASSIFICATION")


    #-------------------------------------- Support Vector Machine ----------------------------------------------------

    #print("SUPPORT VECTOR MACHINE")
    #print("**********************************\n")

    #kFold(D_train, L_train, 5, "SVM", "LIN")
    #kFold(D_train, L_train, 5, "SVM", "POLY")
    #kFold(D_train, L_train, 5, "SVM", "RBF")

    #s.SVM_linear_classifier(DTR, LTR, DTE, LTE, C, K)
    #s.SVM_RBF_classifier(DTR, LTR, DTE, LTE, C, K, gamma)
    #s.SVM_poly_classifier(DTR, LTR, DTE, LTE, C, K, c, d)


    
