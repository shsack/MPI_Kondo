from MatrixProductOperators import dotCavity, bathCouplings
import MPSModule as MPS
import copy
import numpy as np


#
# def trace_environment(density):
#     L = len(density.M)
#
#     for i in range(0, int(L/2) - 1):
#         density.M[i] = np.einsum('lddr->lr', density.M[i])  # trace to dot
#
#     for i in reversed(range(int(L/2) + 1, L)):
#         density.M[i] = np.einsum('lddr->lr', density.M[i])  # trace to cavity
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
#
#     # FIXME: this seems like the issue
#
#     tmp = np.einsum('luvs, sxyr->luxvyr', L, R).squeeze()
#     # tmp = np.einsum('luvs, sxyr->luvxyr', L, R).squeeze() # old version
#
#     tmp_shape = (tmp.shape[0] * tmp.shape[1], tmp.shape[2] * tmp.shape[3])
#
#     return np.reshape(tmp, tmp_shape)  # combine dot and cav
#

def main(epsImp, epsCav, U, omega, Lambda, length, tL, tR, sweeps,D):

    d = 4

    # setting up the Hamiltonian in MPO from
    c = bathCouplings(density='const', N=length, Lambda=Lambda)
    mpo = dotCavity(tL=tL, tR=tR, tBathLeft=c.t, tBathRight=c.t, omega=omega, epsDot=epsImp, epsCav=epsCav,
              epsBathLeft=c.eps, epsBathRight=c.eps, U=U)
    H = MPS.MPO(numberOfSites=len(mpo.Hamiltonian), bondDimension=6, localHilbertSpace=d, maximalBondDimension=D,
              thresholdEntanglement=0., periodic=False)
    H.M = copy.deepcopy([np.transpose(h, (0, 2, 3, 1)) for h in mpo.Hamiltonian])
    H.structuredPhysicalLegs = True

    groundState = MPS.mpsDMRG(H, bondDimension=D, thresholdEntanglement=1e-6)
    groundState.groundState(sweeps=sweeps)

    length_new = len(groundState.M)

    dot_up = MPS.MPO(numberOfSites=length_new, bondDimension=1, localHilbertSpace=d, maximalBondDimension=D,
                    thresholdEntanglement=0., periodic=False)
    dot_up.M = copy.deepcopy([np.transpose(h, (0, 2, 3, 1)) for h in mpo.n_dot_up])
    dot_up.d = [d] * length_new
    dot_up.structuredPhysicalLegs = True


    dot_down = MPS.MPO(numberOfSites=length_new, bondDimension=1, localHilbertSpace=d, maximalBondDimension=D,
                      thresholdEntanglement=0., periodic=False)
    dot_down.M = copy.deepcopy([np.transpose(h, (0, 2, 3, 1)) for h in mpo.n_dot_down])
    dot_down.d = [d] * length_new
    dot_down.structuredPhysicalLegs = True

    cav_up = MPS.MPO(numberOfSites=length_new, bondDimension=1, localHilbertSpace=d, maximalBondDimension=D,
                    thresholdEntanglement=0., periodic=False)
    cav_up.M = copy.deepcopy([np.transpose(h, (0, 2, 3, 1)) for h in mpo.n_cav_up])
    cav_up.d = [d] * length_new
    cav_up.structuredPhysicalLegs = True

    cav_down = MPS.MPO(numberOfSites=length_new, bondDimension=1, localHilbertSpace=d, maximalBondDimension=D,
                      thresholdEntanglement=0., periodic=False)
    cav_down.M = copy.deepcopy([np.transpose(h, (0, 2, 3, 1)) for h in mpo.n_cav_down])
    cav_down.d = [d] * length_new
    cav_down.structuredPhysicalLegs = True

    cov_dot_up_cav_down = np.real(groundState.conjugate() * (dot_up * cav_down) * groundState - \
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

    # Old style of getting the purity --> Too memory intensive
    #
    # density = groundState * groundState.conjugate()
    # density.structurePhysicalLegs()
    # reduced_density = trace_environment(density=density)
    # reduced_density_contracted = contract_in(density=reduced_density)
    #
    # purity = np.real(np.trace(reduced_density_contracted @ reduced_density_contracted))

    # print(purity)


    groundState.makeCanonical('Right')
    groundState.moveGauge(int(length_new / 2), False, False)

    L = groundState.M[int(length_new / 2) - 1]
    R = groundState.M[int(length_new / 2)]


    # My stuff
    total_dens = np.einsum('jik, klt->jilt', L, R)
    total_dens = np.reshape(total_dens, (total_dens.shape[0], total_dens.shape[1] * total_dens.shape[2], total_dens.shape[3]))
    total_dens = np.einsum('ijk, ilk->jl', total_dens, total_dens.conj())
    total_purity = np.real(np.trace(total_dens @ total_dens))

    dot_dens = np.einsum('ijk, ilk->jl', L, L.conj())
    dot_purity = np.real(np.trace(dot_dens @ dot_dens))

    cav_dens = np.einsum('ijk, ilk->jl', R, R.conj())
    cav_purity = np.real(np.trace(cav_dens @ cav_dens))


    # Michael's stuff
    # L = np.einsum('ijk,ilm->jlkm', L, L.conj())
    # R = np.einsum('ijk,lmk->iljm', R, R.conj())
    #
    # dens = np.einsum('ijkl,klmn->imjn', L, R)
    # dummySize = dens.shape[0] * dens.shape[1]
    # dens = np.reshape(dens,(dummySize, dummySize))
    #
    # purity = np.real(np.trace(dens @ dens))
    #
    # print(purity)
    #
    #
    dot_occ = np.real(groundState.conjugate() * (dot_up + dot_down) * groundState)
    cav_occ = np.real(groundState.conjugate() * (cav_up + cav_down) * groundState)
    total_occ = dot_occ + cav_occ

    return correlation, total_purity, dot_purity, cav_purity, total_occ, dot_occ, cav_occ

