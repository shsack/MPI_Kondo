from MatrixProductOperators import dot, bathCouplings
import MPSModule
import copy
import numpy as np


def setup(epsImp, U, V, Lambda, length, D, sweeps):

    c = bathCouplings(density='const', N=length, Lambda=Lambda)
    mpo = dot(V=V, epsImp=epsImp, epsBath=c.eps, U=U, N=length, t=c.t)
    H = MPSModule.MPO(numberOfSites=len(mpo.Hamiltonian), bondDimension=6, localHilbertSpace=4, maximalBondDimension=D,
    thresholdEntanglement=0., periodic=False)
    H.M = copy.deepcopy([np.transpose(h, (0, 2, 3, 1)) for h in mpo.Hamiltonian])
    H.structuredPhysicalLegs = True
    groundState = MPSModule.mpsDMRG(H, bondDimension=D, thresholdEntanglement=1e-6)
    groundState.groundState(sweeps=sweeps)

    return groundState


def main(D, V, epsImp, U, Lambda, length, sweeps):

    groundState = setup(epsImp=epsImp, U=U, V=V, Lambda=Lambda, length=length, D=D, sweeps=sweeps)
    groundState.makeCanonical('Right')
    tmp = groundState.M[0]
    dens = np.einsum('ijk, lmk', tmp, tmp.conj()).squeeze()
    purity = np.real(np.trace(dens @ dens))

    return (purity, )
