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
import matplotlib
import scipy.linalg
import scipy.optimize
import sys
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
import seaborn as sb



import LR as l
import SVM as s
import GAU as g
import support_functions as sf
import preprocessing as p



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
    DTR = p.zero_values(DTR)
    DTE = p.zero_values(DTE)

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
    #--------------------------------------------------PCA----------------------------------------------------------------------
    
    # define dimension wanted to reduce to
    dim = 10

    # UNPROCESSED PCA
    DTR_p = p.pca(DTR, dim)
    #DTR_p = DTR_p[0]
    DTE_p = p.pca(DTE, dim)
    #DTE_p = DTE_p[0]

    # GRASSIANIZED PCA
    DTR_gp = p.pca(DTR_g, dim)
    DTE_gp = p.pca(DTE_g, dim)

    #---------------------------------------------Logistic regression-----------------------------------------------------------
    
    #print("LOGISITC REGRESSION CLASSIFICATION")

    #lamb = [1e-6, 1e-3, 0.1, 1.0, 2.0 , 10.0]
    #l.log_reg_classifier(DTR, LTR, DTE, LTE, lamb)

    #-------------------------------------- Multivariate Gaussian Classifier----------------------------------------------------
    
    print("GAUSSIAN CLASSIFICATION \n")
    priors = [0.5, 0.5]
    #priors = [0.33, 0.67]
    

    print("PCA PROCESSED DATA")
    g.gaussian_classifiers(priors, DTR_p, LTR, DTE_p, LTE)

    #print("UNPROCESSED DATA")
    #g.gaussian_classifiers(priors, DTR, LTR, DTE, LTE)

    print("GAUSSIANIZED DATA")
    g.gaussian_classifiers(priors, DTR_gp, LTR, DTE_gp, LTE)

    #---------------------------------------Mixed Model Gaussian Classifier-----------------------------------------------------

    #print("MIXED MODEL GAUSSIAN CLASSIFICATION")

    #-------------------------------------------------LDA-----------------------------------------------------------------------
    
    #print("LDA CLASSIFICATION")

    #lda.LDA(D_train,L_train)

    #-------------------------------------- Support Vector Machine ----------------------------------------------------

    #clf = s.SVM()
    #clf.fit(D_train, L_train)
    #predictions = clf.predict(L_train)

    #def accuracy(y_true, y_pred):
    #    accuracy = numpy.sum(y_true==y_pred) / len(y_true)
    #    return accuracy*100

    #print("Support Vector Machine")
    #print("Accuracy: ", accuracy(L_test, predictions), "%")



    
