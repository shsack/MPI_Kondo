from MatrixProductOperators import dotCavity, bathCouplings
import MPSModule as MPS
import copy
import numpy as np


def dmrg(epsImp, epsCav):

    # define the simulation parameters
    D = 40
    d = 4
    Lambda = 2.0
    length = 20
    U = 0.5
    omega = 0.025
    tL, tR = 0.05, 0.05

    # setting up the Hamiltonian in MPO from
    c = bathCouplings(density='const', N=length, Lambda=Lambda)
    mpo = dotCavity(tL=tL, tR=tR, tBathLeft=c.t, tBathRight=c.t, omega=omega, epsDot=epsImp, epsCav=epsCav,
              epsBathLeft=c.eps, epsBathRight=c.eps, U=U)
    H = MPS.MPO(numberOfSites=len(mpo.Hamiltonian), bondDimension=6, localHilbertSpace=d, maximalBondDimension=D,
              thresholdEntanglement=0., periodic=False)
    H.M = copy.deepcopy([np.transpose(h, (0, 2, 3, 1)) for h in mpo.Hamiltonian])
    H.structuredPhysicalLegs = True

    groundState = MPS.mpsDMRG(H, bondDimension=D, thresholdEntanglement=1e-6)
    groundState.groundState(sweeps=5)

    length_new = len(groundState.M)

    dot_up =  MPS.MPO(numberOfSites=length_new, bondDimension=1, localHilbertSpace=d, maximalBondDimension=D,
                    thresholdEntanglement=0., periodic=False)
    dot_up.M = copy.deepcopy([np.transpose(h, (0, 2, 3, 1)) for h in mpo.n_dot_up])
    dot_up.d = [d] * length_new
    dot_up.structuredPhysicalLegs = True


    dot_down =  MPS.MPO(numberOfSites=length_new, bondDimension=1, localHilbertSpace=d, maximalBondDimension=D,
                      thresholdEntanglement=0., periodic=False)
    dot_down.M = copy.deepcopy([np.transpose(h, (0, 2, 3, 1)) for h in mpo.n_dot_down])
    dot_down.d = [d] * length_new
    dot_down.structuredPhysicalLegs = True

    cav_up =  MPS.MPO(numberOfSites=length_new, bondDimension=1, localHilbertSpace=d, maximalBondDimension=D,
                    thresholdEntanglement=0., periodic=False)
    cav_up.M = copy.deepcopy([np.transpose(h, (0, 2, 3, 1)) for h in mpo.n_cav_up])
    cav_up.d = [d] * length_new
    cav_up.structuredPhysicalLegs = True

    cav_down =  MPS.MPO(numberOfSites=length_new, bondDimension=1, localHilbertSpace=d, maximalBondDimension=D,
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


    groundState.makeCanonical('Right')
    groundState.moveGauge(int(length_new / 2), False, False)

    L = groundState.M[int(length_new / 2) - 1]
    R = groundState.M[int(length_new / 2)]

    # Michael's stuff
    L = np.einsum('ijk,ilm->jlkm', L, L.conj())
    R = np.einsum('ijk,lmk->iljm', R, R.conj())

    dens = np.einsum('ijkl,klmn->imjn',L,R)
    dummySize = dens.shape[0] * dens.shape[1]
    dens = np.reshape(dens,(dummySize, dummySize))

    # Calculate the observables
    purity = np.real(np.trace(dens @ dens))
    dot_occ = np.real(groundState.conjugate() * (dot_up + dot_down) * groundState)
    cav_occ = np.real(groundState.conjugate() * (cav_up + cav_down) * groundState)
    total_occ = dot_occ + cav_occ

    return correlation, purity, total_occ, dot_occ, cav_occ

