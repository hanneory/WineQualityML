import numpy as np
from matplotlib.axis import XTick
import matplotlib.pyplot as plt
import preprocessing as p

import pandas as pd
import seaborn as sb


def plot_hist(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    hFea = {
        0: 'Fixed acidity',
        1: 'Volatile acidity',
        2: 'Citric acid',
        3: 'Residual sugar',
        4: 'Chlorides',
        5: 'Free sulfur dioxide',
        6: 'Total sulfur dioxide',
        7: 'Density',
        8: 'pH',
        9: 'Sulphates',
        10: 'Alcohol'
    }


    for dIdx in range(11):
        plt.figure()
        plt.xlabel(hFea[dIdx])

        plt.hist(D0[dIdx, :], density = True, alpha = 0.4, label = 'Bad quality')
        plt.hist(D1[dIdx, :], density = True, alpha = 0.4, label = 'Good quality')
        plt.legend()
        plt.tight_layout()
    plt.show()

def plot_general_data():
    
    df = pd.read_csv('Data/Train.txt', header = None)
    df.columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
    
    # CREATE PAIRPLOT
    sb.pairplot(df)
    plt.show()
    plt.savefig('pairplot.pdf')

    # CREATE HEATMAP
    plt.rc ('font', size = 7)
    plt.rc ('xtick', labelsize= 7)
    plt.rc ('ytick', labelsize = 7)
    sb.heatmap(df.corr(), annot=True)
    plt.show()

    # CREATE COMBINED HISTOGRAMS
    plt.rc ('font', size = 7)
    plt.rc ('xtick', labelsize= 7)
    plt.rc ('ytick', labelsize = 7)
    df.hist(bins=25,figsize=(8,8))
    plt.show()


def mrow(V):
    return V.reshape((1, V.size))


def plot_gaus(DTR, LTR, DTE, LTE):

    DTR, DTE = p.gaussianize(DTR, DTE)

    #DTR = DTR.T
    #DTE = DTE.T

    D0 = DTR[:, LTR == 0]
    D1 = DTR[:, LTR == 1]

    hFea = {
        0: 'Fixed acidity',
        1: 'Volatile acidity',
        2: 'Citric acid',
        3: 'Residual sugar',
        4: 'Chlorides',
        5: 'Free sulfur dioxide',
        6: 'Total sulfur dioxide',
        7: 'Density',
        8: 'pH',
        9: 'Sulphates',
        10: 'Alcohol'
    }

    for dIdx in range(11):
        plt.figure()
        plt.xlabel(hFea[dIdx])
        plt.hist(D0[dIdx, :], density = True, alpha = 0.4, label = 'Bad quality')
        plt.hist(D1[dIdx, :], density = True, alpha = 0.4, label = 'Good quality')
        #plt.ylim([0,0.6])
        plt.legend()
        plt.tight_layout() 
        d = dIdx
        plt.savefig('gaus_hist_%d.pdf' % dIdx)
    plt.show()    

    df = pd.DataFrame(DTR.T)
    df.columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    plt.rc ('font', size = 8)
    plt.rc ('xtick', labelsize= 7)
    plt.rc ('ytick', labelsize = 7)
    df.hist(figsize=(8,8))
    plt.show()

    # CREATE HEATMAP
    sb.heatmap(df.corr(), annot=True)
    plt.rc ('font', size = 8)
    plt.rc ('xtick', labelsize= 7)
    plt.rc ('ytick', labelsize = 7)
    plt.savefig('heatmap_gaus.pdf')
    plt.show()

def plot_scatter(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    hFea = {
        0: 'Fixed acidity',
        1: 'Volatile acidity',
        2: 'Citric acid',
        3: 'Residual sugar',
        4: 'Chlorides',
        5: 'Free sulfur dioxide',
        6: 'Total sulfur dioxide',
        7: 'Density',
        8: 'pH',
        9: 'Sulphates',
        10: 'Alcohol'
    }

    for dIdx1 in range (11):
        for dIdx2 in range (11):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(hFea[dIdx1])
            plt.ylabel(hFea[dIdx2])
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = 'Bad Quality')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = 'Good Quality')
            plt.legend()
            plt.tight_layout()
            #plt.savefig('scatter_%d_%d.pdf' % (dIdx1, dIdx2))
        plt.show()

def accuracy(LTE, Lpred):
    return (1 - (np.sum(LTE==Lpred) / len(LTE)))*100

def accuracy_SVM(y_true, y_pred):
    accuracy = np.sum(y_true==y_pred) / len(y_true)
    return (1 - accuracy)*100

#change to column-vector
def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))


