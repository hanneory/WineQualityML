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
import pca
import mvg
import visualize


def mcol(v):
    return v.reshape((v.size, 1))


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

    ## Visualize ##
    #plt.rc ('font', size = 16)
    #plt.rc ('xtick', labelsize=16)
    #plt.rc ('ytick', labelsize = 16)
    #visualize.plot_scatter(D_train, L_train)
    #visualize.plot_hist(D_train, L_train)
    df = pd.read_csv('Data/Train.txt', header = None)
    df.columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
    
    #plt.rc ('font', size = 7)
    #plt.rc ('xtick', labelsize= 7)
    #plt.rc ('ytick', labelsize = 7)
    #print(df.head())
    #print(df.info())
    #print(df.describe())
    #df.hist(bins=25,figsize=(8,8))
    #plt.show()

    #plt.figure(figsize=[10,6])
    #plt.bar(df[11], df[10], color = 'red')
    #plt.xlabel('quality')
    #plt.ylabel('alcohol')
    #plt.show()

    plt.figure(figsize=[10,6])
    sb.heatmap(df.corr(), annot=True)
    plt.show()

    ## Logistic regression ##
    #for lamb in [1e-6, 1e-3, 0.1, 1.0]:
    #    logreg_obj = logistic_regression.logreg_obj_wrap(D_train, L_train, lamb)
    #    _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(D_train.shape[0]+1), approx_grad=True, iprint=1)
    #    _w = _v[0: D_train.shape[0]]
    #    _b = _v[-1]
    #    STE = numpy.dot(_w.T, D_test) + _b
    #    LP = STE > 0
        #print(lamb, _J)
    
    ## PCA ##
    #pca.pca(D_test)

    ## Multivariate Gaussian Classifier ##
    #print(mvg.computeMVG(D_train, L_train, D_test, L_test))



    
