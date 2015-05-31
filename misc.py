##
# Miscellaneous helper functions
##

from numpy import *

def random_weight_matrix(m, n):
    #### YOUR CODE HERE ####
    val=sqrt(6)/sqrt(m+n)
    A0=random.rand(m,n)*2*val-val

    #### END YOUR CODE ####
    assert(A0.shape == (m,n))
    return A0