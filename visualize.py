from random import random
import sys
import numpy
from matplotlib.axis import XTick
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import pandas as pd
import seaborn as sb

def mcol(v):
    return v.reshape((v.size, 1))

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
        #plt.savefig('hist_%d.pdf' % dIdx)
    plt.show()

def mrow(V):
    return V.reshape((1, V.size))


def plot_hist_gaus(D, L):

    df = pd.read_csv('Data/Train.txt', header = None)
    print(df.isnull().sum())
    df.columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
    
    # CREATE PAIRPLOT
    # sb.pairplot(df)
    #plt.show()

    D, test = train_test_split(D, test_size=0.2, random_state=40)

    norm = MinMaxScaler()
    norm_fit = norm.fit(D)
    D = norm_fit.transform(D)
    print(D)

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
        #plt.savefig('hist_%d.pdf' % dIdx)
    #plt.show()

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


