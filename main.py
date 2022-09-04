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

import numpy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb



import LR as l
import SVM as s
import GAU as g
import support_functions as sf
import preprocessing as p
import performance as pf



def shuffle(D, L):
    #rearrange samples
    i = numpy.random.permutation(D.shape[1])
    D = D[:, i]
    L = L[:, i]

    return D, L


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

if __name__ == '__main__':

    #--------------------------------------------------LOAD DATA----------------------------------------------------------------
   
    D_train, L_train = load('Data/Train.txt')
    D_test, L_test = load('Data/Test.txt')

    #D_train = numpy.random.shuffle(D_train)

    #------------------------------------------------VISUALIZATION--------------------------------------------------------------
    
    #sf.plot_scatter(D_train, L_train)
    #sf.plot_gaus(DTR, LTR, DTE, LTE)
    #sf.plot_general_data()

    #------------------------------------------------PREPROCESSING--------------------------------------------------------------
    
    # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    (DTR, LTR), (DTE, LTE) = p.split_db_2to1(D_train, L_train)

    # SHUFFLE DATASET
    # DTR, LTR = shuffle(D_train, L_train)

    # ZERO-VALUE HANDLING
    #DTR = p.zero_values(DTR)
    #DTE = p.zero_values(DTE)

    # GAUSSIANIZATION
    DTR_g, DTE_g = p.gaussianize(DTR, DTE)

    # SPLIT INTO X FOLDS GAUSSIANIZED
    #nFolds = 3
    #DTR_sg = numpy.array_split(DTR_g, nFolds, axis=1)
    #LTR_sg = numpy.array_split(LTR_g, nFolds)

    # SPLIT INTO X FOLDS RAW
    #nFolds = 3
    #DTR_s = numpy.array_split(DTR, nFolds, axis=1)
    #LTR_s = numpy.array_split(LTR, nFolds)


    #---------------------------------------------Logistic regression-----------------------------------------------------------
    
    #print("LOGISITC REGRESSION CLASSIFICATION")
    #print("*************************************")

    #lamb = [1e-6, 1e-3, 0.1, 1.0, 10]
    #l.log_reg_classifier(DTR, LTR, DTE, LTE, lamb)

    #-------------------------------------- Multivariate Gaussian Classifier----------------------------------------------------
    
    #print("GAUSSIAN CLASSIFICATION \n")
    #print("**********************************")

    
    #LPred_GAU = g.gaussian_classifiers("MV",priors, DTR, LTR, DTE, LTE)
    #TODO Create a array of labels and plot all together
    #pf.plot_performance(LPred_GAU , LTE)
    #pf.ROC_plot(LPred_GAU, LTE)

    #minDCF5 = pf.compute_min_DCF(LPred_GAU, LTE, 0.5, 1, 1)
    #print(minDCF5)

    #p.plot_minDCF("LogReg", "Lambda", l, minDCF[0], minDCF[1], minDCF[2])

    
    #LPred_GAU = []
    #print("NON-GAUSSIANIZED DATA \n")
    #print("----------------------------------")
    #print("RAW DATA")
    #g.gaussian_classifiers(DTR, LTR, DTE, LTE)

    #for dim in [10, 9, 8]:
    #    DTR_p = p.pca(DTR, dim)
    #    DTE_p = p.pca(DTE, dim)

    #    print("NR. OF DIMENSIONS IN FEATURE SPACE %d" % dim)
    #    LPred_GAU = g.gaussian_classifiers(DTR_p, LTR, DTE_p, LTE)


    
    #print("GAUSSIANIZED DATA")
    #print("----------------------------------")
    #print("RAW DATA")
    #g.gaussian_classifiers(DTR_g, LTR, DTE_g, LTE)
    
    #for dim in [10, 9, 8]:
    #    DTR_pg = p.pca(DTR_g, dim)
    #    DTE_pg = p.pca(DTE_g, dim)

    #    print("NR. OF DIMENSIONS IN FEATURE SPACE %d" % dim)
    #    g.gaussian_classifiers(DTR_pg, LTR, DTE_pg, LTE)
    
    
    #---------------------------------------Mixed Model Gaussian Classifier-----------------------------------------------------

    #print("MIXED MODEL GAUSSIAN CLASSIFICATION")


    #-------------------------------------- Support Vector Machine ----------------------------------------------------

    print("SUPPORT VECTOR MACHINE")
    print("**********************************\n")

    C = 0.01
    K = 1

    print("LINEAR SVM")
    clf_lin = s.SVM()
    clf_lin.SVM_linear(DTR, LTR, C, K)
    S_lin, LP_lin = clf_lin.predict_lin(DTE)

    minDCF = pf.compute_min_DCF(S_lin, LTE, 0.5, 1, 1)
    print("Error:", sf.accuracy_SVM(LTE, LP_lin), "%")
    print(minDCF)

    print("KERNEL SVM")
    clf_RBF = s.SVM()
    clf_RBF.SVM_RBF(DTR, LTR, C,-2, K)
    S_RBF, LP_RBF = clf_RBF.predict_RBF(DTE)
    minDCF = pf.compute_min_DCF(S_RBF, LTE, 0.5, 1, 1)
    print("Error:", sf.accuracy_SVM(LTE, LP_RBF), "%")
    print(minDCF)




    
