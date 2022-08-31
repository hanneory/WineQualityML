import numpy
from matplotlib.axis import XTick
import matplotlib.pyplot as plt
from sklearn import preprocessing 

import pandas as pd
import seaborn as sb

def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))

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

    # CREATE HEATMA
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


def plot_hist_gaus(D, L):
    bc = preprocessing.PowerTransformer()

    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    D_trans_bc = numpy.zeros([1839,1])

    for dIdx in range(11):

        X0_trans_bc = bc.fit_transform(D0[dIdx,:].reshape(-1,1), D0[dIdx,:].reshape(-1,1))
        X1_trans_bc = bc.fit_transform(D1[dIdx,:].reshape(-1,1), D1[dIdx,:].reshape(-1,1))

        if dIdx == 7: 
            #plt.figure()
            #plt.xlabel(hFea[dIdx])
            #plt.hist(X1_trans_bc, density = True, alpha = 0.4, label = 'Good quality')
            X1_trans_bc = X1_trans_bc * (3e15)

        D_insert = numpy.append(X0_trans_bc, X1_trans_bc)
        D_trans_bc = numpy.append(D_trans_bc, mcol(D_insert), axis=1)

        #plt.figure()
        #plt.xlabel(hFea[dIdx])
        #plt.hist(X0_trans_bc, density = True, alpha = 0.4, label = 'Bad quality')
        #plt.hist(X1_trans_bc, density = True, alpha = 0.4, label = 'Good quality')
        #plt.ylim([0,0.6])
        #plt.legend()
        #plt.tight_layout() 
    D_trans_bc = numpy.delete(D_trans_bc, 0 , axis=1)

    df = pd.DataFrame(D_trans_bc)
    df.columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    plt.rc ('font', size = 7)
    plt.rc ('xtick', labelsize= 7)
    plt.rc ('ytick', labelsize = 7)
    df.hist(figsize=(8,8))
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


