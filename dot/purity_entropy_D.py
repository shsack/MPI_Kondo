from MatrixProductOperators import dot, bathCouplings
import MPSModule
import copy
import numpy as np
from scipy.linalg import logm


def setup(epsImp, U, V, Lambda,length, D, sweeps):

    bath = bathCouplings(density='const', N=length, Lambda=Lambda)
    dot_ = dot(t=bath.t, epsImp=epsImp, epsBath=bath.eps, V=V, U=U, N=length)
    anderson = MPSModule.MPO(numberOfSites=length, bondDimension=6, localHilbertSpace=4, maximalBondDimension=D,
    thresholdEntanglement=0., periodic=False)
    anderson.M = copy.deepcopy([np.transpose(h, (0, 2, 3, 1)) for h in dot_.Hamiltonian])
    anderson.structuredPhysicalLegs = True
    groundState = MPSModule.mpsDMRG(anderson, bondDimension=D, thresholdEntanglement=1-6)
    groundState.groundState(sweeps=sweeps)

    return groundState


def main(V, D):

    U = 0.5
    epsImp = U/2
    Lambda = 2.0
    length = 20
    sweeps = 10

    groundState = setup(epsImp=epsImp, U=U, V=V, Lambda=Lambda, length=length, D=D, sweeps=sweeps)
    groundState.makeCanonical('Right')
    tmp = groundState.M[0]
    dens = np.einsum('ijk, lmk', tmp, tmp.conj()).squeeze()
    purity = np.real(np.trace(dens @ dens))
    entropy = -np.real(np.trace(dens @ logm(dens)))

    return purity, entropy
