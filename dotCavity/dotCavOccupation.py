import sys
sys.path.insert(0, '..')
from MatrixProductOperators import dotCavity, bathCouplings
import MPSModule
import copy
import numpy as np


# define the simulation parameters
D = 40
d = 4
Lambda = 2.0
length = 20
U = 0.5


def trace_environment(density):
    L = len(density.M)

    for i in range(0, int(L/2) - 1):
        density.M[i] = np.einsum('lddr->lr', density.M[i]) # trace to dot

    for i in reversed(range(int(L/2) + 1, L)):
        density.M[i] = np.einsum('lddr->lr', density.M[i]) # trace to cavity

    return density


def contract_in(density):
    length = len(density.M)

    for i in range(0, int(length/2) - 2):
        density.M[i + 1] = np.einsum('ij, jk->ik', density.M[i], density.M[i + 1]) # contract to dot

    for i in reversed(range(int(length/2) + 2, length)):
        density.M[i - 1] = np.einsum('ij, jk->ik', density.M[i - 1], density.M[i]) # contract to cavity

    L = np.einsum('ls, sudr->ludr', density.M[int(length/2) - 2], density.M[int(length/2) - 1])
    R = np.einsum('luds, sr->ludr', density.M[int(length/2)], density.M[int(length/2) + 1])

    # tmp = np.einsum('luvs, sxyr->luxvyr', L, R).squeeze()
    tmp = np.einsum('luvs, sxyr->luvxyr', L, R).squeeze(
    tmp_shape = (tmp.shape[0] * tmp.shape[1], tmp.shape[2] * tmp.shape[3])
def main(epsImp, epsCav):
    # setting up the Hamiltonian in MPO from
    c = bathCouplings(density='const', N=length, Lambda=Lambda)
    mpo = dotCavity(tL=0.01, tR=0.01, tBathLeft=c.t, tBathRight=c.t, omega=0.1, epsDot=epsImp, epsCav=epsCav,
              epsBathLeft=c.eps, epsBathRight=c.eps, U=0.5)
    H = MPSModule.MPO(numberOfSites=len(mpo.Hamiltonian), bondDimension=6, localHilbertSpace=4, maximalBondDimension=D,
                             thresholdEntanglement=0., periodic=False)
    H.M = copy.deepcopy([np.transpose(h, (0, 2, 3, 1)) for h in mpo.Hamiltonian])
    H.structuredPhysicalLegs = True
    groundState = MPSModule.mpsDMRG(H, bondDimension=D, thresholdEntanglement=1e-6)
    groundState.groundState(sweeps=5)

    length_new = len(groundState.M)

    dot_up =  MPSModule.MPO(numberOfSites=length_new, bondDimension=1, localHilbertSpace=4, maximalBondDimension=D,
                              thresholdEntanglement=0., periodic=False)
    dot_up.M = copy.deepcopy([np.transpose(h, (0, 2, 3, 1)) for h in mpo.n_dot_up])
    dot_up.d = [d] * length_new
    dot_up.structuredPhysicalLegs = True


    dot_down =  MPSModule.MPO(numberOfSites=length_new, bondDimension=1, localHilbertSpace=4, maximalBondDimension=D,
                              thresholdEntanglement=0., periodic=False)
    dot_down.M = copy.deepcopy([np.transpose(h, (0, 2, 3, 1)) for h in mpo.n_dot_down])
    dot_down.d = [d] * length_new
    dot_down.structuredPhysicalLegs = True

    cav_up =  MPSModule.MPO(numberOfSites=length_new, bondDimension=1, localHilbertSpace=4, maximalBondDimension=D,
                              thresholdEntanglement=0., periodic=False)
    cav_up.M = copy.deepcopy([np.transpose(h, (0, 2, 3, 1)) for h in mpo.n_cav_up])
    cav_up.d = [d] * length_new
    cav_up.structuredPhysicalLegs = True
                                          


















                                                                                                    1,5           Top

