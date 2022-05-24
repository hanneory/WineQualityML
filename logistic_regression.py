import numpy

#change to column-vector
def mcol(v):
    return v.reshape((v.size, 1))

def logreg_obj_wrap(DTR, LTR, l):
    Z = LTR * 2.0 - 1.0
    M = DTR.shape[0]
    def logreg_obj(v):
        w = mcol(v[0:M])
        b = v[-1]
        S = numpy.dot(w.T, DTR) + b
        cxe = numpy.logaddexp(0, -S*Z).mean()
        return cxe + 0.5*l*numpy.linalg.norm(w)**2
    return logreg_obj