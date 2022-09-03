import numpy as np
import scipy.special

import GAU as g
import support_functions as sp

#GMM = function that comptes the likelihood for each sample

def logpdf_GAU_ND_OPT(X, mu, C):
    c = g.logpdf_GAU_ND(X,mu, C)

    Y = []
    for i in range(X.shape[1]):
        x = X[:, i:i+1]
        res = c + -0.5 * np.dot((x-mu).T, np.dot(np.linalg.inv(C), (x-mu)))  
        Y.append(res)

    #TODO finne ut hva denne funksjonen er
    Y = [g.logpdf_GAU_ND(X[:, i:i+1, mu, C]) for i in range(X.shape[i])]
    return np.array(Y).ravel()

def GMM_ll_perSample(X, gmm): #logpdf_GMM

    S = np.zeros((len(gmm), X.shape[1]))

    for g in range(len(gmm)):
        S[g, :] = logpdf_GAU_ND_OPT(X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0])
    
    return scipy.special.logsumexp(S, axis=0)

def compute_LLR(D, gmm0, gmm1):
    logD0 = GMM_ll_perSample(D, gmm0)
    logD1 = GMM_ll_perSample(D, gmm1)

    llr = logD1 - logD0
    return llr

def GMM_EM(X, gmm):

    llNew = None
    llOld = None

    G = len(gmm)
    N = X.shape[1]

    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ = np.zeros((G, N))
        
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND_OPT(X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0])
        
        SM = scipy.special.logsumexp(SJ, axis=0)
        llNew = SM.sum()/N
        P = np.exp(SJ-SM)
        gmmNew = []
        
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (sp.mrow(gamma)*X).sum(1)
            S = np.dot(X, (sp.mrow(gamma)*X).T)
            w = Z/N
            mu = sp.mcol(F/Z)
            sig = S/Z - np.dot(mu, mu.T)
            gmmNew.append((w, mu, sig))
        
        gmm = gmmNew
        print(llNew)
    print(llNew-llOld)
    return gmm