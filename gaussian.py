from distutils.debug import DEBUG
import scipy
import numpy as np

priors = [0.33, 0.67]

def vcol(V):
    return V.reshape((V.size, 1))

def vrow(V):
    return V.reshape((1, V.size))

def create_diagonal_matrix(M, mat_size):

    id_matrix = np.identity(mat_size)

    return M*id_matrix


# LAB SPESIFIC FUNCTIONS
def empirical_mean(X):
    return vcol(X.mean(1))

def empirical_covariance(X):
        
        mu = vcol(X.mean(1))
        xc = X - vcol(mu)
        xcxct = np.dot(xc, xc.T)  / X.shape[1]
    
        return xcxct

def class_parameters(D, L , i):

    SC = empirical_covariance(D[:, L == i]) #calculate only for relevant data
    UC = empirical_mean(D[:, L == i])

    return UC, SC

def logpdf_GAU_ND0(X, mu, C):

    M, num_cols = X.shape
    P = np.linalg.inv(C)

    const = -((M/2)*np.log(2*np.pi)) 
    const = const - 0.5*np.linalg.slogdet(P)[1] 

    Y = []
    
    for i in range(num_cols):
        # X[:, i] gives us the ith column
        Xi = X[:, i:i+1]

        logNi = const + -0.5*np.dot( (Xi - mu).T , np.dot(P, (Xi - mu)))
        Y.append(logNi)

    # The sigma computations are costly, maybe calculate them before entering the loop
    # Even better is doing all the constant calculations first
    return np.array(Y).ravel() # 1D floating point

#ADDED
def logpdf_GAU_ND(X, mu, C):
    P = np.linalg.inv(C)
    return -0.5*X.shape[0]*np.log(np.pi*2) + 0.5*np.linalg.slogdet(P)[1] - 0.5*(np.dot(P, (X-mu))*(X-mu)).sum(0)

def ML_GAU(D):
    mu = vcol(D.mean(1))
    C = np.dot(D-mu, (D-mu).T)/float(D.shape[1])
    return mu, C


def GAU(D_train, L_train, D_test, L_test):
    h = {}

    for i in [0, 1]:
        DX = D_train[:, L_train == i]
        mu, C = ML_GAU(DX)
        h[i] = (mu, C)

    #we now have one table for each class

    SJoint = np.zeros((2, D_test.shape[1]))
    logSJoint = np.zeros((2, D_test.shape[1]))

    classPriors = priors

    for i in [0,1]:
        #compute class conditional densities
        mu, C = h[i]
        SJoint[i, :] = np.exp(logpdf_GAU_ND(D_test, mu, C).ravel()) * classPriors[i]
        logSJoint[i, :] = logpdf_GAU_ND(D_test, mu, C).ravel() * np.log(classPriors[i])

    SMarginal = SJoint.sum(0)
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)

    Post1 = SJoint / vrow(SMarginal)
    #not log
    logPost = logSJoint - vrow(logSMarginal)
    Post2 = np.exp(logPost)
    #log

    #print this
    LPred1 = Post1.argmax(axis=0)
    accuracy = (L_test == LPred1).sum() / L_test.size
    errors = 1 - accuracy

    print("GAUSSIAN CLASSIFIER")
    print("Accuracy: ", accuracy*100, "%")

    






# CLASSIFIERS
def Naive_bayes_gaussian_classifier(classes, DTE):

     # MAKE DIAGONAL MATRIX OF COV. MATRIX TO IMPLEMENT NAIVE BAYES CLASSIFIER
    for i in [0,1]:
        _, S = classes[i]
        size, _ = S.shape
        classes[i][1] = create_diagonal_matrix(S, size)

    # her kunne vi byttet ut 3 med en mer generisk måte å finne antall klasser på 
    ScoreJoint = np.zeros((2, DTE.shape[1]))
    logScoreJoint = np.zeros((2, DTE.shape[1]))

    # This is our prior prediciton, which is that there are
    # equaly many samples belonging to each class
    classPriors = priors

    # PERFORM INTERFERENCE ON THE TEST SAMPLE
    for i in [0,1]:
        U, S = classes[i]

        logDen = logpdf_GAU_ND(DTE, U, S)
        logJointDen = logDen + np.log(classPriors[i])
        
        densities = np.exp(logDen).ravel()
        jointDen = densities*classPriors[i]
        
        ScoreJoint[i, :] = jointDen 
        logScoreJoint[i, :] = logJointDen

        # log densities are the second part of the lab

    # CALCULATE THE MARGINAL SCORE
    SMarginal = ScoreJoint.sum(0)
    logScoreMarginal = scipy.special.logsumexp(logScoreJoint, axis = 0)

    # COMPUTE POSTERIOR PROBABILITY
    posterior =ScoreJoint/vrow(SMarginal) 
    logPosterior = np.exp(logScoreJoint - vrow(logScoreMarginal))

    # GET ONE PREDICTION PER SAMPLE
    prediction = posterior.argmax(0)
    logPrediction = logPosterior.argmax(0)

    return prediction, logPrediction

def Tied_Covariance_Gaussian_Classifier(classes, DTE):
    
    # her kunne vi byttet ut 3 med en mer generisk måte å finne antall klasser på 
    ScoreJoint = np.zeros((2, DTE.shape[1]))
    logScoreJoint = np.zeros((2, DTE.shape[1]))

    # This is our prior prediciton, which is that there are
    # equaly many samples belonging to each class
    classPriors = priors

    #DENNNE KAN BLI FEIL
    tied_cov = np.zeros((DTE.shape[0],DTE.shape[0]))

    # CREATE TIED COVARIANCE MATRIX
    for i in [0,1]:
        _, S = classes[i]
        tied_cov = tied_cov + (2 * S)
        tied_cov = tied_cov / 2

    # PERFORM INTERFERENCE ON THE TEST SAMPLE
    for i in [0,1]:
        U, _ = classes[i]

        logDen = logpdf_GAU_ND(DTE, U, tied_cov)
        logJointDen = logDen + np.log(classPriors[i])
        
        densities = np.exp(logDen).ravel()
        jointDen = densities*classPriors[i]
        
        ScoreJoint[i, :] = jointDen 
        logScoreJoint[i, :] = logJointDen

        # log densities are the second part of the lab

    # CALCULATE THE MARGINAL SCORE
    SMarginal = ScoreJoint.sum(0)
    logScoreMarginal = scipy.special.logsumexp(logScoreJoint, axis = 0)

    # COMPUTE POSTERIOR PROBABILITY
    posterior =ScoreJoint/vrow(SMarginal) 
    logPosterior = np.exp(logScoreJoint - vrow(logScoreMarginal))

    # GET ONE PREDICTION PER SAMPLE
    prediction = posterior.argmax(0)
    logPrediction = logPosterior.argmax(0)

    return prediction, logPrediction

def Multivariant_Gaussian_Classifier(classes, DTE):

     # her kunne vi byttet ut 3 med en mer generisk måte å finne antall klasser på 
    ScoreJoint = np.zeros((2, DTE.shape[1]))
    logScoreJoint = np.zeros((2, DTE.shape[1]))

    # This is our prior prediciton, which is that there are
    # equaly many samples belonging to each class
    classPriors = priors

    # PERFORM INTERFERENCE ON THE TEST SAMPLE
    for i in [0,1]:
        U, S = classes[i]

        logDen = logpdf_GAU_ND(DTE, U, S)
        logJointDen = logDen + np.log(classPriors[i])
        
        densities = np.exp(logDen).ravel()
        jointDen = densities*classPriors[i]
        
        ScoreJoint[i, :] = jointDen 
        logScoreJoint[i, :] = logJointDen

        # log densities are the second part of the lab

    # CALCULATE THE MARGINAL SCORE
    SMarginal = ScoreJoint.sum(0)
    logScoreMarginal = scipy.special.logsumexp(logScoreJoint, axis = 0)

    # COMPUTE POSTERIOR PROBABILITY
    posterior =ScoreJoint/vrow(SMarginal) 
    logPosterior = np.exp(logScoreJoint - vrow(logScoreMarginal))

    # GET ONE PREDICTION PER SAMPLE
    prediction = posterior.argmax(0)
    logPrediction = logPosterior.argmax(0)

    return prediction, logPrediction



def gaussians(DTR, LTR, DTE, LTE):
    

    # FIND THE EMPIRICAL MEAN AND COVARIANCE
    U0, S0 = class_parameters(DTR, LTR, 0)
    U1, S1 = class_parameters(DTR, LTR, 1)

    MVG_classes = [[U0, S0], [U1, S1]]
    NB_classes  = [[U0, S0], [U1, S1]]
    TC_classes  = [[U0, S0], [U1, S1]]

    # MULTIVARIANT EVALUATION
    MVG_pred, MVG_logPred = Multivariant_Gaussian_Classifier(MVG_classes, DTE)
    MVG_solution = (LTE == MVG_pred)

    MVG_accuracy = MVG_solution.sum() / LTE.size
    MVG_errors = 1 - MVG_accuracy

    print("Multivariant Gaussian Classifier")
    print("Accuracy:", MVG_accuracy*100, "%")
    print("Error:   ", MVG_errors*100, "% \n")

    # NAIVE BAYESIAN EVALUATION
    NB_pred, NB_logPred = Naive_bayes_gaussian_classifier(NB_classes, DTE)
    NB_solution = (LTE == NB_pred)

    NB_accuracy = NB_solution.sum() / LTE.size
    NB_errors = 1 - NB_accuracy

    print("Naive Bayes Gaussian Classifier")
    print("Accuracy:", NB_accuracy*100, "%")
    print("Error:   ", NB_errors*100, "% \n")

    # TIED COVARIANCE EVALUATION
    TC_pred, TC_logPred = Tied_Covariance_Gaussian_Classifier(TC_classes, DTE)
    TC_solution = (LTE == TC_pred)

    TC_accuracy = TC_solution.sum() / LTE.size
    TC_errors = 1 - TC_accuracy

    print("Tied Covariance Gaussian Classifier")
    print("Accuracy:", TC_accuracy*100, "%")
    print("Error:   ", TC_errors*100, "% \n")


    

