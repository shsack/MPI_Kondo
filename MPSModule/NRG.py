import numpy as np
import scipy
#from mpsMethods import mpsDMRG
#import copy
from multiprocessing import Pool, set_start_method, cpu_count
from itertools import repeat
#import numba



def matEl(L, W, M1, M2, R):
    # TODO rename
    # TODO documentation
    # TODO tensordot implementation
    # TODO check if a complex conjugation is needed on M1 or M2
    return np.einsum('aik, abst, tkl, sij, bjl', L, W, M1, M2, R, dtype=np.complex128, optimize=True)

def optimizeSingleSiteFull(L, R, W, M, computeEigenVectors):
    # TODO move to DMRG file
    # TODO documentation
    # TODO: implement with tensordot
    dim = M.shape[0] * M.shape[1] * M.shape[2]
    H = np.einsum('aik, abst, bjl->tklsij', L, W, R, dtype=np.complex128, optimize=True)
    H = np.reshape(H, (dim, dim))

    if computeEigenVectors:
        # TODO write the following without lambda
        eigvals, eigvecs = scipy.linalg.eigh(H)
        reshape = lambda m: np.reshape(m, (M.shape[0], M.shape[1], M.shape[2]))
        reshaped_eigvecs = (reshape(eigvecs[:, index]) for index in range(dim))

        return eigvals, reshaped_eigvecs

    return scipy.linalg.eigvalsh(H)

def spectrum(run):

    set_start_method('spawn')  # fixes some strange bug on Mac OS which causes the process to dispatch

    Lenv, Renv = run.constructFullEnvironments()
    run.indexOrderInplace('stefan')
    Hamiltonian = run.MPO.M
    spectrum = []

    # TODO: fix constructFullEnvironment such that Renv is in the right order and Pool can be used

    for l, r, h, m in zip(Lenv, Renv[::-1], Hamiltonian, run.M):
        spectrum.append(optimizeSingleSiteFull(l, r, h, m, False))
    #
    # with Pool(processes=cpu_count()) as pool:
    #     spectrum = pool.starmap(optimizeSingleSiteFull, zip(Lenv, Renv, Hamiltonian, run.M, repeat(False)))

    run.indexOrderInplace('DdD')

    return spectrum


def spectralFunction_at_one_site(groundState, Hamiltonian, annihilationMPO, L, R):

    energies, states = optimizeSingleSiteFull(Hamiltonian, groundState, L, R, True)
    matrixElements = [(matEl(L, annihilationMPO, groundState, state, R),
                       matEl(L, annihilationMPO, state, groundState, R)) for state in states]

    return matrixElements, energies


def spectralFunction(run, annihilationMPO):

    set_start_method('spawn')  # fixes some strange bug on Mac OS which causes the process to dispatch
    Lenv, Renv = run.constructFullEnvironments()
    run.indexOrderInplace('stefan')
    Hamiltonian = run.MPO.M
    spectrum, matrixElements = [], []

    for l, r, h, m in zip(Lenv, Renv[::-1], Hamiltonian, run.M):

        e, m = optimizeSingleSiteFull(l, r, h, m, True)
        spectrum.append(e), matrixElements.append(m)


    #Lenv = construct_left_env(Hamiltonian, groundState)
    #Renv = construct_right_env(Hamiltonian, groundState)

    # with Pool(processes=cpu_count()) as pool:
    #     matrixElements, spectrum = \
    #         zip(*pool.starmap(spectralFunction_at_one_site, zip(groundState, Hamiltonian, annihilationMPO, Lenv, Renv)))

    return matrixElements, spectrum