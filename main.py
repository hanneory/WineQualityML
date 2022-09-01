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



import logistic_regression
import preprocessing as p
import mvg
import visualize as v
import lda
import gaussian as g

def mcol(v):
    return v.reshape((v.size, 1))


def shuffle(d):
    #rearrange samples
    numpy.random.shuffle(d)

    #split into groups
    arr = numpy.array_split(d, 40)
    return arr


#load file and and changes row and column. Can't get sklearn to work on mac. 
def load(fname):
    DList = []
    labelList = []

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:11]
                attrs = mcol(numpy.array([float(i) for i in attrs])) 
                name = line.split(',')[-1].strip()
                DList.append(attrs)
                labelList.append(float(name))
            except: #har except fordi det kan hende f.eks siste linja i fila ikkje er lik resten så funksjonane over fungerer ikkje.
                pass
    return numpy.hstack(DList), numpy.array(labelList, dtype = numpy.int32) #Liste med vectors som eg stacke horisontalt som vil gi meg ei matrise. Har også list of labels som lager label-array.

if __name__ == '__main__':
    D_train, L_train = load('Data/Train.txt')
    D_test, L_test = load('Data/Test.txt')

    #------------------------------------------------PREPROCESSING--------------------------------------------------------------
    
    # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    (DTR, LTR), (DTE, LTE) = p.split_db_2to1(D_train, L_train)

    # ZERO-VALUE HANDLING
    

    # GAUSSIANIZATION
    DTR_g = p.gaussianize(DTR, LTR)
    DTE_g = p.gaussianize(DTE, LTE)

    #------------------------------------------------VISUALIZATION--------------------------------------------------------------
    #v.plot_scatter(D_train, L_train)
    #v.plot_gaus(D_train, L_train)
    #v.plot_general_data()


    #---------------------------------------------Logistic regression-----------------------------------------------------------
    
    #for lamb in [1e-6, 1e-3, 0.1, 1.0]:
    #    logreg_obj = logistic_regression.logreg_obj_wrap(D_train, L_train, lamb)
    #    _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(D_train.shape[0]+1), approx_grad=True, iprint=1)
    #    _w = _v[0: D_train.shape[0]]
    #    _b = _v[-1]
    #    STE = numpy.dot(_w.T, D_test) + _b
    #    LP = STE > 0
        #print(lamb, _J)
    
    #--------------------------------------------------PCA----------------------------------------------------------------------
    
    #p.pca(D_train)

    #-------------------------------------- Multivariate Gaussian Classifier----------------------------------------------------
    
    print("UNPROCESSED DATA")
    g.gaussians(DTR, LTR, DTE, LTE)
    print("GAUSSIANIZED DATA")
    g.gaussians(DTR_g, LTR, DTE_g, LTE)

    #-------------------------------------------------LDA-----------------------------------------------------------------------
    #lda.LDA(D_train,L_train)



    
