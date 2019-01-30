import numpy as np
import copy
import numbers
import bisect
from .generic import directSum
from .generic import intArrayfromIntOrList
from .generic import isint
from .generic import checkIntorIntArray


def MPOproperties(MPOlist):

    L = len(MPOlist)

    shapes = np.asarray( [np.shape(M) for M in MPOlist])
    d1 = shapes[:,2]
    d2 = shapes[:,3]
    assert d1==d2, 'invalid MPO'
    d = d1

    D1 = shapes[:,0]
    D2 = shapes[:,1]
    assert np.roll(D2, 1) == D1, 'invalid MPO'
    D = D21

    periodic = True
    if D[-1] == 1: periodic = False

    return {
        'numberOfSites': L,
        'localHilbertSpace': d,
        'bondDimension': D,
        'periodic': periodic
    }



def MPOconstructor(MPOlist):

    kwargs = MPOproperties((MPOlist))
    output = MPO(**kwargs)
    output.indexOrderInplace('DDd')
    output._MPO__structureInplace()
    output.M = MPOlist
    output._MPO__flattenInplace()
    output.indexOrderInplace('DdD')

    return output

def indexPermutation(old = 'DdD', new = 'DDd'):
    """
    A method to find the tensor permutations between two representations
    not for external use
    :param old: string representing the old format
    :param new: string representing the new format
    :return:  tuple, representing the permutation
    """
    if old == new: return (0,1,2)
    if old == 'DdD' or new == 'DdD':
        if new == 'DDd' or old == 'DDd': return (0,2,1)
        if new == 'dDD' or old == 'dDD': return (1,0,2)
    if (old == 'DDd' or new == 'DDd') and (old == 'dDD' or new == 'dDD'): return (2,1,0)

    assert False, 'invalid permutations'



class MP:
    # TODO docstring
    """

    """

    # TODO add compression function
    # TODO documentation
    # CC: we have a lot of different multiplications, can we make an overview?
    # CC: do we want docstring for every member function? Don't know what the policy is here...

    def __init__(self, **kwargs):
        # pop the arguments and check there are none left
        self.L = kwargs.pop('numberOfSites', 5)
        self.d = kwargs.pop('localHilbertSpace', 2)
        self.D = kwargs.pop('bondDimension', 10)
        self.periodic = kwargs.pop('periodic', False)
        self.threshold = kwargs.pop('thresholdEntanglement', 0)
        self.maxD = kwargs.pop('maximalBondDimension', 20)
        self.physicalLegs = kwargs.pop('physicalLegs', 1)
        if kwargs:
            raise TypeError('got unexpected keyword argument(s), {0}'.format(str(kwargs.keys())))

        # sanity checks
        assert isint(self.L), 'non integer chain length'
        assert isint(self.physicalLegs)
        assert isinstance(self.periodic, bool)
        assert self.threshold >= 0, 'invalid threshold entanglement'
        assert self.maxD >= 0, 'invalid maximal bond dimension'
        # the local Hilbert space may be an int or a list of ints, stored as
        # a list of ints
        self.d = intArrayfromIntOrList(self.d, self.L)  # physical leg size
        self.dd = copy.deepcopy(self.d) ** self.physicalLegs  # product of all physical legs
        # the bond dimension may be an int or a list of ints, stored as
        # a list of ints
        self.D = intArrayfromIntOrList(self.D, self.L)
        if not self.periodic: self.D[-1] = 1

        # initiate
        self.reset()  # sets the following four parameters to 0 or defaults
        self.M  # the tensors representing the MPS
        self.S  # the list of singular values
        self.structuredPhysicalLegs  # are the legs flattened or structured?
        self.gauge  # the location of the gauge
        self.scalar  # a scalar with which should be taken care of in addition or multiplication
        self.order # a string which tracks the order of indices in the M matrices
        return

    def reset(self):
        self.structuredPhysicalLegs = False
        self.gauge = None
        # self.M = [np.zeros((self.D[i - 1], self.dd[i], self.D[i]), dtype='complex')
        #           for i in range(self.L)]
        tmp = []
        for i in range(self.L):
            tmp.append(np.zeros((self.D[i - 1], self.dd[i], self.D[i])))
        self.M = tmp
        self.S = [np.zeros(self.D[i]) for i in range(self.L)]
        self.scalar = 1.0
        self.order = 'DdD'
        return

    def kwargs(self): # don't overload this in a daughter class, should probably be a private function
        return {
            'numberOfSites': self.L,
            'localHilbertSpace': self.d,
            'bondDimension': self.D,
            'periodic': self.periodic,
            'thresholdEntanglement': self.threshold,
            'maximalBondDimension': self.maxD,
            'physicalLegs': self.physicalLegs
        }

    def adjustTruncationKwargs(self, other):
        return {
            'thresholdEntanglement': min(self.threshold, other.threshold),
            'maximalBondDimension': max(self.maxD, other.maxD)
        }

    def indexOrderInplace(self, order = 'DdD'): # should be hidden maybe?
        if self.order == order: return
        self.__flattenInplace()
        permutations = indexPermutation(self.order, order)
        for i in range(self.L):
            self.M[i] = np.transpose(self.M[i], permutations)
        self.order = order
        if self.structuredPhysicalLegs: self.__structureInplace()

        return

    def flattenPhysicalLegs(self):
        # CC: combines two physical legs of an operator to one physical leg
        # CC: why not: if type(self)==MPS: return
        self.__flattenInplace()
        self.structuredPhysicalLegs = False
        return

    def __flattenInplace(self):  # does not change the flag
        for i in range(self.L):
            if self.order == 'DdD':
                shape = np.array((self.D[i - 1], self.dd[i], self.D[i]))
            if self.order == 'dDD':
                shape = np.array((self.dd[i], self.D[i - 1], self.D[i]))
            if self.order == 'DDd':
                shape = np.array((self.D[i - 1], self.D[i], self.dd[i]))
            self.M[i] = np.reshape(self.M[i], shape)
        return

    def structurePhysicalLegs(self):
        # CC: splits physical legs for an operator
        # CC: why not: if type(self)==MPS: return
        self.__structureInplace()
        self.structuredPhysicalLegs = True
        return

    def __structureInplace(self):  # does not change the flag
        for i in range(self.L):
            if self.order == 'DdD':
                shape = [self.D[i - 1]] + [self.d[i]] * self.physicalLegs + [self.D[i]]
            if self.order == 'dDD':
                shape = [self.d[i]] * self.physicalLegs + [self.D[i - 1]] + [self.D[i]]
            if self.order == 'DDd':
                shape =  [self.D[i - 1]] + [self.D[i]] + [self.d[i]] * self.physicalLegs
            self.M[i] = np.reshape(self.M[i], shape)
        return

    def norm(self):  # highly dependent on the MP structure
        return 1

    def normalise(self):
        N = self.norm()
        N = np.power(N, 0.5)
        self.__scalarMultiplicationInplace(N)
        return

    def setRandomState(self):
        self.M = [np.random.rand(self.D[i - 1], self.dd[i], self.D[i])/self.D[i] + 1j/self.D[i] for i in range(self.L)]
        self.normalise()
        return

    def setProductState(self, state):
        state = np.asarray(state)
        assert state.dtype == 'int', 'setProductState takes a list of ints of length L as its argument'
        assert state.ndim == 1, 'setProductState takes a list of ints of length L as its argument'
        assert len(state) == self.L, 'setProductState takes a list of ints of length L as its argument'
        self.D = np.ones(self.L, dtype='int')
        self.maxD = 1
        self.reset()
        self.__structureInplace()
        for site, value in enumerate(state):
            idx = [0] + [value] * self.physicalLegs + [0]
            self.M[site][tuple(idx)] = 1
            self.S[site] += 1

        self.__flattenInplace()
        return

    def __conjugateInplace(self):  # use with care! Slow, replaces all matrices
        self.scalar = np.conjugate(self.scalar)
        for i in range(self.L):
            self.M[i] = self.M[i].conjugate()
        return

    def conjugate(self):
        new = copy.deepcopy(self)
        new.__conjugateInplace()
        return new

    def compatibilityCheck(self, other):
        assert np.all(self.d == other.d), 'these objects do not match'
        assert self.periodic == other.periodic, 'these objects do not match'

    def spawnAdjustedCopy(self, other):
        self.compatibilityCheck(other)
        new = copy.deepcopy(self)
        new.maxD = max(self.maxD, other.maxD)
        new.threshold = min(self.threshold, other.threshold)
        return new

    def __add__(self, other):
        assert type(self) is type(other), 'these objects cannot be added'
        if self.structuredPhysicalLegs != other.structuredPhysicalLegs:
            print('Warning: you are adding two MPs with different representations')

        new = self.spawnAdjustedCopy(other)
        new.D = self.D + other.D
        new.maxD = self.maxD + other.maxD
        new.scalar = 1
        self.__flattenInplace()
        self.__distributedScalarMultiplication(self.scalar, sites=0)
        other.__flattenInplace()
        other.__distributedScalarMultiplication(self.scalar, sites=0)

        for i in range(self.L):
            new.M[i] = directSum(
                self.M[i].transpose(1, 0, 2),
                other.M[i].transpose(1, 0, 2)
            ).transpose(1, 0, 2)

        # CC: for non-periodic MP, matrices of first site are row, matrices of last site column vectors
        if not self.periodic:
            new.M[0] = np.expand_dims(np.sum(new.M[0], axis=0), axis=0)
            new.M[-1] = np.expand_dims(np.sum(new.M[-1], axis=-1), axis=-1)
            new.D[-1] = 1

        self.__distributedScalarMultiplication(1 / self.scalar, sites=0)
        other.__distributedScalarMultiplication(1 / self.scalar, sites=0)
        if new.structuredPhysicalLegs: new.__structureInplace()
        if self.structuredPhysicalLegs: self.__structureInplace()
        if other.structuredPhysicalLegs: other.__structureInplace()
        return new

    def __rsub__(self, other):
        # CC: what's this?
        return other + (-1) * self

    def __sub__(self, other):
        # CC: what's this?
        return self + (-1) * other

    def makeScalarUnity(self, sites = 0):
        # CC: do we have need to have this structure with public function calling private function?
        self.__distributedScalarMultiplication(self.scalar, sites)  # only on zeroth site
        self.scalar = 1

    def __distributedScalarMultiplication(self, scalar, sites=None):  # communism
        if sites == None: sites = range(self.L)
        if scalar == None:
            scalar = copy.deepcopy(self.scalar)
            self.scalar = 1
        sites = checkIntorIntArray(sites)
        assert len(sites) <= self.L
        N = scalar ** (1 / len(sites))
        for i in range(self.L):
            self.M[i] = self.M[i] * N

    def __scalarMultiplicationInplace(self, scalar):  # capitalist distribution
        self.scalar *= scalar

    def __rmul__(self, other):
        if isinstance(other, numbers.Number): return self * other
        raise TypeError('Cannot multiply these objects')

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            N = other ** (1 / self.L)
            new = copy.deepcopy(self)
            new.__scalarMultiplicationInplace(other)
            return new
        raise TypeError('Cannot multiply these objects')

    def makeCanonical(self, direction='Right', truncate = False, normalise = False):
        assert direction in ['Right', 'Left'], 'invalid direction'
        if direction == 'Right':
            tmp = 1
        else:
            tmp = 0
        direction = (2 * tmp) - 1
        if self.periodic:
            self.moveGauge(direction * self.L, truncate, normalise)
            self.M[self.gauge] = self.M[self.gauge] / np.sqrt(self.norm)
            return
        self.gauge = tmp - 1
        self.moveGauge(direction * self.L, truncate, normalise)
        self.moveGauge(-direction * self.L, truncate, normalise)
        return

    def moveGauge(self, distance, truncate, normalise):
        # CC: do we have need to have this structure with public function calling private function?
        distance = np.asarray(distance).flatten()
        assert distance.dtype == np.int64 and np.shape(distance)[0] == 1, 'invalid displacement'
        distance = distance[0]
        self.__flattenInplace()
        for i in range(np.abs(distance)):
            self.__moveGaugeOnce(distance, truncate, normalise)

        if self.structuredPhysicalLegs: self.__structureInplace()
        return

    def __moveGaugeOnce(self, direction, truncate, normalise):
        # internal use with care, no safety checks
        # positive direction is right, negative direction is left
        idx = (self.gauge + self.L) % self.L  # such that negative gauges work
        # Check which direction we are moving
        permutation = (0, 1, 2)
        newIdx = (idx + 1) % self.L  # make sure it won't fail for periodic guys
        entropyIdx = idx
        boundary = idx == self.L - 1 and not self.periodic
        if direction < 0:
            permutation = (2, 1, 0)  # take the transpose to move right
            newIdx = idx - 1
            entropyIdx = idx - 1
            boundary = idx == 0 and not self.periodic

        site = self.M[idx].transpose(permutation)
        chi0, d, chi2 = np.shape(site)

        left, right = chi0 * d, chi2  # as we work in transposed basis always move right
        # CC: left, right are not used
        u, s, vh = np.linalg.svd(site.reshape(chi0 * d, chi2), full_matrices=False)

        if truncate:
            D = bisect.bisect_left(-s, -self.threshold) # CC: what does bisect do?
            D = min(D, self.maxD)
            u = u[:, :D]
            norm = np.linalg.norm(s[:D])
            s = s[:D]
            if normalise: s = s / norm
            vh = vh[:D, :]

        self.D[entropyIdx] = np.shape(s)[0]
        self.S[entropyIdx] = s
        self.M[idx] = u.reshape(chi0, d, self.D[entropyIdx]).transpose(permutation)

        if not boundary:
            newSite = self.M[newIdx].transpose(permutation)
            self.M[newIdx] = np.einsum(
                'i,ik,klm->ilm', s, vh, newSite
            ).transpose(permutation)
            self.gauge = (newIdx + self.L) % self.L  # leave a positive gauge
        else:
            self.__scalarMultiplicationInplace(np.einsum('i,ij->j', s, vh)[0])

        return

    def __str__(self):
        output = """Matrix Product object
        type = {0}
        number of sites = {1:d}
        physical dimensions = {2}
        bond dimensions = {3}""".format(
            str(type(self)),
            self.L,
            str(self.d),
            str(self.D)
        )
        return output


class MPS(MP):
    # TODO add measure
    def __init__(self, **kwargs
                 ):
        """
        Initialise an MPS with zero tensors
        """
        kwargs['physicalLegs'] = 1
        super().__init__(**kwargs)
        return

    def reset(self):  # is called by base __init__
        super().reset()
        self.cc = False  # False is a ket, True is a bra
        return

    def norm(self):
        if not self.cc: return np.abs(self.conjugate() * self)
        if self.cc: return np.abs(self * self.conjugate())

    def normalise(self):
        N = self.norm()
        N = np.power(N, 1 / (2 * self.L))
        for i in range(self.L):
            self.M[i] = self.M[i] / N
        return

    def compress(self, initialGuess):  # this function is what you need Stefan
        # TODO call super().compress (compress is the same for MPS and MPO)
        return

    def __conjugateInplace(self):  # use with care! Slow, replaces all matrices
        for i in range(self.L):
            self.M[i] = self.M[i].conjugate()
        self.cc = not self.cc
        return

    def conjugate(self):
        new = copy.deepcopy(self)
        new.__conjugateInplace()
        return new

    def measure(self, operator):
        return self.conjugate()*operator*self

    def var(self, MPO):
        return self.measure(MPO*MPO) - self.measure(MPO) ** 2

    def __add__(self, other):
        assert type(self) == type(other), 'Cannot these objects'
        assert self.cc == other.cc, 'Cannot add a bra to a ket'
        return super().__add__(other)

    def __innerProduct(self, other):  # unsafe! use with care!
        tmp = np.einsum('ijk,ljm->ilkm', self.M[0], other.M[0])
        for i in range(1, self.L):
            tmp = np.einsum('ilkm,kjn,mjo->ilno', tmp, self.M[i], other.M[i])
        return np.einsum('ijij', tmp) * self.scalar * other.scalar

    def __outerProduct(self, other):  # unsafe! use with care!
        kwargs = self.kwargs()
        kwargs.update(self.adjustTruncationKwargs(other))
        kwargs['physicalLegs'] = 2
        kwargs['bondDimension'] = self.D * other.D
        kwargs['maximalBondDimension'] = self.maxD * other.maxD
        new = MPO(**kwargs)
        new.structurePhysicalLegs()
        for i in range(self.L):
            new.M[i] = np.einsum('ijk,lmn->iljmkn', self.M[i], other.M[i])
            #new.S[i] = np.kron(self.S[i], other.S[i])
        new.flattenPhysicalLegs()
        new.scalar = self.scalar * other.scalar
        return new

    def __mul__(self, other):

        try:
            return super().__mul__(other)  # if other is a scalar this should work
        except TypeError:
            None

        self.compatibilityCheck(other)
        if isinstance(other, MPS):
            assert self.cc != other.cc, 'Cannot multiply two bras or two kets'
            if self.cc and not other.cc:  # take inner product
                return self.__innerProduct(other)

            if not self.cc and other.cc:  # take outer product (make MPO)
                return self.__outerProduct(other)

        if isinstance(other, MPO):
            assert self.cc, 'kets cannot act from the left on operators'
            new = self.__unsafeMul(other)
            new.cc = True
            return new

        raise TypeError('Cannot multiply these objects')

    def __unsafeMul(self, other):  # unsafe multiplication, assumes smart user
        kwargs = self.kwargs()
        kwargs.update(self.adjustTruncationKwargs(other))
        kwargs['bondDimension'] = self.D * other.D
        new = MPS(**kwargs)
        other._MP__structureInplace()
        for i in range(self.L):
            new.M[i] = np.einsum('ijk,ljmn->ilmkn', self.M[i], other.M[i])
            new.M[i] = np.reshape(new.M[i],
                                  (self.D[i - 1] * other.D[i - 1],
                                   self.d[i],
                                   self.D[i] * other.D[i]))

        if not other.structuredPhysicalLegs: other._MP__flattenInplace()
        new.scalar = self.scalar * other.scalar
        return new

    def __str__(self):
        string = super().__str__()
        additional = '''
        complex conjugate = {0}'''.format(str(self.cc))
        return string + additional


class MPO(MP):
    # TODO add trace and measure
    def __init__(self, **kwargs):
        kwargs['physicalLegs'] = 2
        super().__init__(**kwargs)
        self.reset()
        return

    def makeIdentity(self):
        self.D = np.ones(self.L, dtype='int')
        self.reset()
        for site in range(self.L):
            localIdentity = np.ones(self.d[site], dtype='complex')
            localIdentity = np.diag(localIdentity)
            localIdentity = np.expand_dims(localIdentity, 0)
            localIdentity = np.expand_dims(localIdentity, -1)
            self.M[site] = copy.deepcopy(localIdentity)

    def HermitianConjugate(self):
        new = self.conjugate()
        new.__transposeInplace()
        return new

    def makeHermitian(self):
        new = self + self.HermitianConjugate()
        new = 0.5 * new
        return new

    def __transposeInplace(self):  # use with care! Fast, provides a new view
        self._MP__structureInplace()
        for i in range(self.L):
            self.M[i] = self.M[i].transpose((0, 2, 1, 3))
        if not self.structuredPhysicalLegs: self._MP__flattenInplace()
        return

    def transpose(self):
        new = copy.deepcopy(self)
        return new.__transposeInplace()

    def __radd__(self, other):
        return self + other

    def __add__(self, other):

        if isinstance(other, numbers.Number):
            kwargs = self.kwargs()
            kwargs['bondDimension'] = 1
            identity = MPO(**kwargs)
            identity.makeIdentity()
            return identity * other + self

        return super().__add__(other)

    def __operatorMul(self, other):
        kwargs = self.kwargs()
        kwargs.update(self.adjustTruncationKwargs(other))
        kwargs['bondDimension'] = self.D * other.D
        new = MPO(**kwargs)
        new.scalar = self.scalar * other.scalar
        self._MP__structureInplace()
        other._MP__structureInplace()
        for i in range(self.L):
            new.M[i] = np.einsum('ijkl,mkop->imjolp', self.M[i], other.M[i])
        if new.structuredPhysicalLegs:
            new._MP__structureInplace()
        else:
            new._MP__flattenInplace()
        if not self.structuredPhysicalLegs: self._MP__flattenInplace()
        if not other.structuredPhysicalLegs: other._MP__flattenInplace()
        return new

    def __mul__(self, other):
        try:
            return super().__mul__(other)  # if other is a scalar this will work
        except TypeError:
            None

        self.compatibilityCheck(other)

        if type(other) == MPS:
            assert not other.cc, 'operators cannot act on bras from the left'
            self.__transposeInplace()
            new = other._MPS__unsafeMul(self)
            new.cc = False
            self.__transposeInplace()
            return new

        if type(other) == MPO:
            return self.__operatorMul(other)

        raise TypeError('Cannot multiply these objects')

    # def norm(self): # do we want a norm?
