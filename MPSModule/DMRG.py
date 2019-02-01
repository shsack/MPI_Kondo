import scipy.sparse.linalg
from .MatrixProducts import *
#from .MatrixProducts import MPOconstructor
#from .MatrixProducts import MPOproperties

import time
import numpy as np
import scipy.sparse as sparse

import sys

def contract_from_right(W, A, F, B):
    # TODO docstring
    # TODO camelcase
    """

    :param W:
    :param A:
    :param F:
    :param B:
    :return:
    """
    tmp = np.tensordot(A, F, axes=(2, 1))
    tmp = np.tensordot(tmp, W, axes=((0, 2), (2, 1)))
    return np.transpose(np.tensordot(tmp, B.conj(), axes=((1, 3), (2, 0))), axes=(1, 0, 2))


def contract_from_left(W, A, E, B):
    # TODO docstring
    # TODO camelcase
    """

    :param W:
    :param A:
    :param E:
    :param B:
    :return:
    """

    tmp = np.tensordot(A, E, axes=(1, 1))
    tmp = np.tensordot(tmp, W, axes=((0, 2), (2, 0)))
    return np.transpose(np.tensordot(tmp, B.conj(), axes=((1, 3), (1, 0))), axes=(1, 0, 2))

def optimizeSingleSiteSparse(L, R, W, M, k = 1):
    # TODO docstring
    """
    optimizes the MPS at one site and returns it without proper SVD procedure
    :param L:
    :param R:
    :param W:
    :param M:
    :param k:
    :param mode:
    :return:
    """

    dim = M.shape[0] * M.shape[1] * M.shape[2]

    # print(np.shape(L), np.shape(R), np.shape(W), np.shape(M))

    def matvec(x):
        # calculate the action of the matrix on a vector

        v = np.tensordot(L, np.reshape(x, (M.shape[0], M.shape[1], M.shape[2])), axes=(1, 1))
        v = np.tensordot(v, W, axes=((0, 2), (0, 2)))
        v = np.transpose(np.tensordot(v, R, axes=((1, 2), (1, 0))), axes=(1, 0, 2))

        return np.reshape(v, -1)


    # define the linear operator to be diagonalized, this is faster than actually storing the matrix
    H = sparse.linalg.LinearOperator((dim, dim), matvec, dtype=np.complex128)


    # solve the eigenvalue problem while using the current MPS at site as an initial guess for the eigenvector

    a = sparse.linalg.eigsh(H, k=k, which='SA', v0=np.reshape(M, -1))

    # FIXME: mp gets stuck here

    return a


class mpsDMRG(MPS):

    def __init__(self, MPO, **kwargs):

        """

            Minimize the energy E = <psi/H/psi> by optimizing wave function /psi> given in MPS form, /psi> = MPS
            After the ground state MPS is found, expectation values of operators given in MPO form can be calculated

            -----------

            * Parameters:

            - MPO (list of complex rank=3 tensors):

                The Hamiltonian given in MPO form.

            - D (int):

                The bond dimension, which is given by d*(number of distinct operators in the Hamiltonian + 2).
                Larger bond dimensions can be used, which enlarges the dimension of the eigenvalue problem,
                slows down the simulation but enhances the accuracy of the DMRG simulation. DMRG is exact for
                D -> inf.

            ---------

            * Complexity:

                The cost of the simulation is dominated by the (d*D^2, d*D^2) eigenvalue problem, a fast
                Lanczos diagonalization scales as O(d*D^2), the diagonalization has to be repeated N (=chain length) times
                so for one swipe we have:

                           Complexity ~ O(d * D^2 * N)

        """

        if type(MPO) == list:
            MPO = MPOconstructor(MPO)

        # pop the arguments and check there are none left
        D = kwargs.pop('bondDimension', 10)
        self.MPO = MPO  # store the passed MPO

        tmp = MPO.kwargs()
        # print(kwargs)
        tmp.update(
            {
                'bondDimension': D,
                'maximalBondDimension': D,
                'physicalLegs': 1
            }
        )

        tmp.update(kwargs)

        kwargs = tmp

        super().__init__(**kwargs)
        return

    def indexOrderInplace(self, order='DdD'):
        if order == 'stefan':
            self.MPO.indexOrderInplace('DDd')
            super().indexOrderInplace('dDD')
            return
        if order == 'DdD':
            self.MPO.indexOrderInplace('DdD')
            super().indexOrderInplace('DdD')
            return

    def constructFullEnvironments(self):

        # TODO: Use Gamma, M notation to make this easier

        #Lenv = [np.ones((1, 1, 1), dtype=np.complex128)]
        #for i in range(self.L - 1):
        #    Lenv.append(contract_from_left(self.MPO.M[i], self.M[i], Lenv[-1], self.M[i]))

        gauge = self.gauge
        self.makeCanonical('Right')
        Renv = self.construct_right_env()
        self.makeCanonical('Left')
        Lenv = self.construct_left_env()
        self.moveGauge(gauge - self.L, False, False)
        return Lenv, Renv

    def construct_right_env(self):
        # TODO: Use Gamma, M notation to make this easier
        # TODO camelCase
        """
        computes the right environment for the first sweep

        """
        self.indexOrderInplace('stefan')

        R = [np.ones((1, 1, 1), dtype=np.complex128)]
        for i in range(self.L - 1, 0, -1):
            R.append(contract_from_right(self.MPO.M[i], self.M[i], R[-1], self.M[i]))
        self.indexOrderInplace('DdD')
        return R

    def construct_left_env(self):  # only works properly on
        # TODO: Use Gamma, M notation to make this easier
        # TODO camelCase
        self.indexOrderInplace('stefan')

        L = [np.ones((1, 1, 1), dtype=np.complex128)]
        for i in range(self.L - 1):
            L.append(contract_from_left(self.MPO.M[i], self.M[i], L[-1], self.M[i]))
        self.indexOrderInplace('DdD')
        return L

    def getMaxD(self):
        return max(self.D)

    def optimizeOneSite(self):

        """optimizes the MPS at one site and returns it without proper SVD procedure"""

        self.indexOrderInplace('stefan')

        i =self.gauge
        W = self.MPO.M[i]
        M = self.M[i]
        L = self.Lenv[-1]
        R = self.Renv[-1]

        eigval, eigvec = optimizeSingleSiteSparse(L, R, W, M, k=1)

        # reshape the resulting eigenvector into MPS form
        self.M[i] = np.reshape(eigvec[:, 0], (M.shape[0], M.shape[1], M.shape[2]))

        self.indexOrderInplace('DdD')

        return

    def singleSiteDMRG(self, sweeps):

        """

        Performs the single site DMRG algorithm by swiping through the MPS chain
        and optimizing the MPSs individually

        """
        self.Lenv = [np.ones((1, 1, 1), dtype=np.complex128)]
        self.Renv = self.construct_right_env()

        for sweep in range(sweeps):
            for i in range(0, self.L - 1):
                self.optimizeOneSite()
                self.moveGauge(1, True, True)
                self.indexOrderInplace('stefan')
                self.Lenv.append(contract_from_left(self.MPO.M[i], self.M[i], self.Lenv[-1], self.M[i]))
                self.indexOrderInplace('DdD')
                self.Renv.pop()

            for i in range(self.L - 1, 0, -1):
                self.optimizeOneSite()
                self.moveGauge(-1, True, True)
                self.indexOrderInplace('stefan')
                self.Renv.append(contract_from_right(self.MPO.M[i], self.M[i], self.Renv[-1], self.M[i]))
                self.indexOrderInplace('DdD')
                self.Lenv.pop()

        return


    def groundState(self, sweeps):

        self.setRandomState()
        self.normalise()
        self.makeCanonical('Right')
        self.singleSiteDMRG(sweeps)
        return self.M
