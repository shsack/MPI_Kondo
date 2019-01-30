import sys
sys.path.insert(0, '..')
from MatrixProductOperators import dotCavity, bathCouplings
import MPSModule
import copy
import numpy as np


# define the simulation parameters
D = 200
d = 4
Lambda = 2.0
length = 20
U = 0.5

#
# def trace_environment(density):
#     L = len(density.M)
#
#     for i in range(0, int(L/2) - 1):
#         density.M[i] = np.einsum('lddr->lr', density.M[i]) # trace to dot
#
#     for i in reversed(range(int(L/2) + 1, L)):
#         density.M[i] = np.einsum('lddr->lr', density.M[i]) # trace to cavity
#
#     return density
#
#
# def contract_in(density):
#     length = len(density.M)
#
#     for i in range(0, int(length/2) - 2):
#         density.M[i + 1] = np.einsum('ij, jk->ik', density.M[i], density.M[i + 1]) # contract to dot
#
#     for i in reversed(range(int(length/2) + 2, length)):
#         density.M[i - 1] = np.einsum('ij, jk->ik', density.M[i - 1], density.M[i]) # contract to cavity
#
#     L = np.einsum('ls, sudr->ludr', density.M[int(length/2) - 2], density.M[int(length/2) - 1])
#     R = np.einsum('luds, sr->ludr', density.M[int(length/2)], density.M[int(length/2) + 1])
#
#     # tmp = np.einsum('luvs, sxyr->luxvyr', L, R).squeeze()
#     tmp = np.einsum('luvs, sxyr->luvxyr', L, R).squeeze()
#
#     tmp_shape = (tmp.shape[0] * tmp.shape[1], tmp.shape[2] * tmp.shape[3])
#
#     return  np.reshape(tmp, tmp_shape) # combine dot and cav


def main(epsImp, epsCav):
    # setting up the Hamiltonian in MPO from
    c = bathCouplings(density='const', N=length, Lambda=Lambda)
    mpo = dotCavity(tL=0.05, tR=0.05, tBathLeft=c.t, tBathRight=c.t, omega=0.025, epsDot=epsImp, epsCav=epsCav,
              epsBathLeft=c.eps, epsBathRight=c.eps, U=U)
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

    cav_down =  MPSModule.MPO(numberOfSites=length_new, bondDimension=1, localHilbertSpace=4, maximalBondDimension=D,
                              thresholdEntanglement=0., periodic=False)
    cav_down.M = copy.deepcopy([np.transpose(h, (0, 2, 3, 1)) for h in mpo.n_cav_down])
    cav_down.d = [d] * length_new
    cav_down.structuredPhysicalLegs = True

    cov_dot_up_cav_down =  np.real(groundState.conjugate() * (dot_up * cav_down) * groundState - \
           (groundState.conjugate() * dot_up * groundState) * (groundState.conjugate() * cav_down * groundState))

    std_dot_up = np.sqrt(np.real(groundState.conjugate() * (dot_up * dot_up) * groundState -
                    (groundState.conjugate() * dot_up * groundState)**2))

    std_cav_down = np.sqrt(np.real(groundState.conjugate() * (cav_down * cav_down) * groundState -
                    (groundState.conjugate() * cav_down * groundState)**2))


    cov_dot_down_cav_up =  np.real(groundState.conjugate() * (dot_down * cav_up) * groundState - \
           (groundState.conjugate() * dot_down * groundState) * (groundState.conjugate() * cav_up * groundState))

    std_dot_down = np.sqrt(np.real(groundState.conjugate() * (dot_down * dot_down) * groundState -
                    (groundState.conjugate() * dot_down * groundState)**2))

    std_cav_up = np.sqrt(np.real(groundState.conjugate() * (cav_up * cav_up) * groundState -
                    (groundState.conjugate() * cav_up * groundState)**2))

    correlation = cov_dot_up_cav_down / (std_dot_up * std_cav_down) + cov_dot_down_cav_up / (std_dot_down * std_cav_up)
    #
    # density = groundState * groundState.conjugate()
    # density.structurePhysicalLegs()
    # reduced_density = trace_environment(density=density)
    # reduced_density_contracted = contract_in(density=reduced_density)
    # purity = np.real(np.trace(reduced_density_contracted @ reduced_density_contracted))

    groundState.makeCanonical('Right')
    groundState.moveGauge(int(length_new / 2), False, False)

    L = groundState.M[int(length_new / 2) - 1]
    R = groundState.M[int(length_new / 2)]

    # Michael's garbage
    L = np.einsum('ijk,ilm->jlkm',L,L.conj())
    R = np.einsum('ijk,lmk->iljm',R,R.conj())

    dens = np.einsum('ijkl,klmn->imjn',L,R)
    dummySize = dens.shape[0]*dens.shape[1]
    dens = np.reshape(dens,(dummySize,dummySize))
    #

    #dens = np.einsum('jik, klt->jilt', L, R)
    #dens = np.reshape(dens, (dens.shape[0], dens.shape[1] * dens.shape[2], dens.shape[3]))

    #dens = np.einsum('ijk, ilk->jl', dens, dens.conj())
    purity = np.real(np.trace(dens @ dens))


    dot_occ = np.real(groundState.conjugate() * (dot_up + dot_down) * groundState)
    cav_occ = np.real(groundState.conjugate() * (cav_up + cav_down) * groundState)

    total_occ = dot_occ + cav_occ

    return correlation, purity, total_occ, dot_occ, cav_occ



if __name__ == '__main__':

    epsDot = float(sys.argv[1])

    epsCav = float(sys.argv[2])

    correlation, purity, total_occ, dot_occ, cav_occ = main(epsDot, epsCav)

    f = open("correlation_purity_occupation.txt", "a")
    f.write(str(epsDot) + " " + str(epsCav) + " " + str(correlation) + " " + str(purity) + " " + str(total_occ) + " " + str(dot_occ) + " " + str(cav_occ) + "\n")
    f.close()


