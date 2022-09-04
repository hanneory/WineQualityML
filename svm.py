## support vector machine 

## Need to take care of:
## Hyperparameters, how to select good values -> try different values to figure this out. Cross-validiation can help. 
## We need to take care in selecting the appropriate kernel. Kernels often include hyper-parameters (gamma for RBF). Cross-validation can help selection of good values.
## We need to select good values of the coefficient C. Again, cross-validation can help. 
## Can be usefu to scale and whiten the data
## Consider problem with no bias.

import support_functions as sp
import numpy as np
import scipy.optimize
from itertools import repeat

#often useful to centralize the data and normalize the variance
#has no probabilistic interpretation - either postprosess or use validation set.
# if we have very unbalanced datasets it can be useful to rebalance the sets to aviod it minimizing errors in one class
# TODO compare linear svm with logreg

#RFB has hyperparameter gamma
#poly has hyperparameter d

#to add the kernel we change the claculation of xi*xj, here denoted H to be K(xi*xj)

class SVM:

    def SVM_RBF(self, DTR, LTR, C, gamma, K = 1):
        #this simulates the effect of a bias
        DTREXT = np.vstack([DTR, np.ones((1, DTR.shape[1]))*K])

        self.Z = np.zeros(LTR.shape)
        self.Z[LTR == 1] = 1
        self.Z[LTR == 0] = -1


        self.DTR = DTR
        self.LTR = LTR
        self.K = K
        self.C = C
        self.gamma = gamma

        # KERNEL
        Dist = np.zeros([DTR.shape[1], DTR.shape[1]])
        for i in range(DTR.shape[1]):
            for j in range(DTR.shape[1]):
                xi = DTR[:, i]
                xj = DTR[:, j]
                Dist[i, j] = np.linalg.norm(xi-xj)**2
        H = np.exp(-gamma*Dist) + K # adding K to compensate for bias

        #to get zi*zj*H*H.T
        H = sp.mcol(self.Z) * sp.mrow(self.Z) * H

        def JDual(a):
            Ha = np.dot(H, sp.mcol(a))
            aHa = np.dot(sp.mrow(a), Ha)
            al = a.sum()
            return -0.5 * aHa.ravel() + al, -Ha.ravel() + np.ones(a.size)
        
        #optimize instead of minimize
        def Ldual(a):
            loss, grad = JDual(a)
            return -loss, - grad

        def JPrimal(w):
            S = np.dot(sp.mrow(w), DTREXT)
            loss = np.maximum(np.zeros(S.shape), 1 - self.Z * S).sum()
            return 0.5 * np.linalg.norm(w)**2 + C * loss

        #factor should be set to 1
        aStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(Ldual, np.zeros(DTR.shape[1]), bounds= [(0,C)]*DTR.shape[1], factr= 0.0, maxiter=100000, maxfun=100000)
        #aStar is the score for the test sample

        print(_x)
        print(_y)

        wStar = np.dot(DTREXT, sp.mcol(aStar) * sp.mcol(self.Z))
        self.w = wStar

        print(JPrimal(wStar))
        print(JDual(aStar)[0])


    def SVM_linear(self, DTR, LTR, C, K = 1):

        #this simulates the effect of a bias
        DTREXT = np.vstack([DTR, np.ones((1, DTR.shape[1]))])

        Z = np.zeros(LTR.shape)
        Z[LTR == 1] = 1
        Z[LTR == 0] = -1

        self.K = K
        self.C = C

        H = np.dot(DTREXT.T, DTREXT)
        #Dist = sp.mcol((DTR ** 2).sum(0)) + sp.mrow((DTR**2).sum(0)) - 2 * np.dot(DTR.T, DTR)
        #H = np.exp(-Dist)
        H = sp.mcol(Z) * sp.mrow(Z) * H
    

        def JDual(a):
            Ha = np.dot(H, sp.mcol(a))
            aHa = np.dot(sp.mrow(a), Ha)
            al = a.sum()
            return -0.5 * aHa.ravel() + al, -Ha.ravel() + np.ones(a.size)
        
        def Ldual(a):
            loss, grad = JDual(a)
            return -loss, - grad

        def JPrimal(w):
            S = np.dot(sp.mrow(w), DTREXT)
            loss = np.maximum(np.zeros(S.shape), 1 - Z * S).sum()
            return 0.5 * np.linalg.norm(w)**2 + C * loss

        #factor should be set to 1
        aStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(Ldual, np.zeros(DTR.shape[1]), bounds= [(0,C)]*DTR.shape[1], factr= 0.0, maxiter=100000, maxfun=100000)
        
        #print("_x", _x)
        #print("_y", _y)

        wStar = np.dot(DTREXT, sp.mcol(aStar) * sp.mcol(Z))

        self.w = wStar

        return

    def predict_lin(self, DTE):
        DTE = np.vstack([DTE, np.zeros(DTE.shape[1])+self.K])
        S = np.dot(self.w.T, DTE)
        LP = 1*(S > 0)
        LP[LP == 0] = -1
             

        #print("Score: ", S)
        #print("LP: ", LP)
        return S, LP

    def predict_RBF(self, DTE):
        H = np.zeros((self.DTR.shape[0], DTE.shape[0]))
        for i in range(self.DTR.shape[0]):
            for j in range(DTE.shape[0]):
                H[i,j]=np.exp(-self.gamma*(np.linalg.norm(self.DTR[:,i]-self.DTR[:,j])**2))+self.K**2
        S=np.sum(np.dot((self.w*self.LTR).reshape(1, self.DTR.shape[1]), H), axis=0)
        LP = 1*(S > 0)
        LP[LP == 0] = -1  
        return S, LP

