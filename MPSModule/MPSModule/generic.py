# generic functions, should think of where this should go
import numpy as np

def directSum(A,B):
    
    assert np.shape(A)[:-2] == np.shape(B)[:-2], 'all but the last two dimensions of A & B must be the same'
    assert A.dtype == B.dtype, ('A ({0:s}) & B ({1:s}) arrays must be of the same type'
                                .format(str(A.dtype),str(B.dtype)))

    shapeA    = np.asarray( np.shape(A) )
    shapeB    = np.asarray( np.shape(B) )
    shapeC    = np.append( shapeA[:-2], shapeA[-2:] + shapeB[-2:] )
    shapeTmpA = np.append( np.product(shapeA[:-2]), shapeA[-2:] )
    shapeTmpB = np.append( np.product(shapeB[:-2]), shapeB[-2:] )
    shapeTmpC = np.append( np.product(shapeC[:-2]), shapeC[-2:] )
    
    tmpA = np.reshape(A, shapeTmpA)
    tmpB = np.reshape(B, shapeTmpB)
    tmpC = np.zeros(shapeTmpC, dtype = A.dtype)
    tmpC[:,:shapeA[-2],:shapeA[-1]] = tmpA
    tmpC[:,shapeA[-2]:,shapeA[-1]:] = tmpB

    return np.reshape(tmpC, shapeC)

def checkIntorIntArray(A): # flattens array
    A = np.asarray(A).flatten()
    assert A.dtype == np.int64, 'array is not integers'
    return A

def intArrayfromIntOrList(A, length): # flattens array
    A = np.asarray(A).flatten()
    assert A.dtype == np.int64, 'array is not integers'
    if np.shape(A)[0] == 1:
        return np.repeat(A, length)
    else:
        assert np.shape(A)[0] == length, 'invalid size'
        return A
    
def isint(i): # should check that i is an int and return it if correct
    return i