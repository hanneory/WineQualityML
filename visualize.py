import sys
import numpy
from matplotlib.axis import XTick
import matplotlib.pyplot as plt

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
        plt.hist(D0[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'Bad quality')
        plt.hist(D1[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'Good quality')
        plt.legend()
        plt.tight_layout()
        plt.savefig('hist_%d.pdf' % dIdx)
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
            plt.savefig('scatter_%d_%d.pdf' % (dIdx1, dIdx2))
        plt.show()


