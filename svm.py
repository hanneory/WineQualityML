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

#often useful to centralize the data and normalize the variance
#has no probabilistic interpretation - either postprosess or use validation set.
# if we have very unbalanced datasets it can be useful to rebalance the sets to aviod it minimizing errors in one class
# TODO compare linear svm with logreg

#RFB has hyperparameter gamma
#poly has hyperparameter d

#to add the kernel we change the claculation of xi*xj, here denoted H to be K(xi*xj)

def SVM_linear(DTR, LTR, C, K = 1):
    #this simulates the effect of a bias
    DTREXT = np.vstack([DTR, np.ones((1, DTR.shape[1]))*K])

    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    # LINEAR
    #H = np.dot(DTREXT.T, DTREXT)
    #Dist = sp.mcol((DTR ** 2).sum(0)) + sp.mrow((DTR**2).sum(0)) - 2 * np.dot(DTR.T, DTR)
    #H = np.exp(-gamma*Dist)

    # KERNEL
    Dist = np.zeros(DTR.shape[0], DTR.shape[0])
    for i in range(DTR.shape[0]):
        for j in range(DTR.shape[1]):
            xi = DTR[:, i]
            xj = DTR[:, j]
            Dist[i, j] = np.linalg.norm(xi-xj)**2
    H = np.exp(-gamma*Dist) + K # adding K to compensate for bias

    #to get zi*zj*H*H.T
    H = sp.mcol(Z) * sp.mrow(Z) * H

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
        loss = np.maximum(np.zeros(S.shape), 1 - Z * S).sum()
        return 0.5 * np.linalg.norm(w)**2 + C * loss

    #will comverge
    #factor should be set to 1
    aStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(Ldual, np.zeros(DTR.shape[1]), bounds= [(0,C)]*DTR.shape[1], factr= 0.0, maxiter=100000, maxfun=100000)
    #aStar is the score for the test sample

    print(_x)
    print(_y)

    wStar = np.dot(DTREXT, sp.mcol(aStar) * sp.mcol(Z))

    print(JPrimal(wStar))
    print(JDual(aStar)[0])

def SVM_classifier(DTR, LTR, C, K = 1):

    #this simulates the effect of a bias
    DTREXT = np.vstack([DTR, np.ones((1, DTR.shape[1]))])

    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    Dist = sp.mcol((DTR ** 2).sum(0)) + sp.mrow((DTR**2).sum(0)) - 2 * np.dot(DTR.T, DTR)
    H = np.exp(-Dist)
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
    
    print(_x)
    print(_y)

    wStar = np.dot(DTREXT, sp.mcol(aStar) * sp.mcol(Z))

    print(JPrimal(wStar))
    print(JDual(aStar)[0])

    return


class SVM:
    def __init__(self, learning_rate=1e-3, lambda_param=1e-2, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def _init_weights_bias(self, X):
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0

    def _get_cls_map(self, y):
        return np.where(y <= 0, -1, 1)

    def _satisfy_constraint(self, x, idx):
        linear_model = np.dot(x, self.w) + self.b 
        return self.cls_map[idx] * linear_model >= 1
    
    def _get_gradients(self, constrain, x, idx):
        if constrain:
            dw = self.lambda_param * self.w
            db = 0
            return dw, db
        
        dw = self.lambda_param * self.w - np.dot(self.cls_map[idx], x)
        db = - self.cls_map[idx]
        return dw, db
    
    def _update_weights_bias(self, dw, db):
        self.w -= self.lr * dw
        self.b -= self.lr * db
    
    def fit(self, X, y):
        self._init_weights_bias(X)
        self.cls_map = self._get_cls_map(y)

        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                constrain = self._satisfy_constraint(x, idx)
                dw, db = self._get_gradients(constrain, x, idx)
                self._update_weights_bias(dw, db)
    
    def predict(self, X):
        estimate = np.dot(X, self.w) + self.b
        prediction = np.sign(estimate)
        return np.where(prediction == -1, 0, 1)