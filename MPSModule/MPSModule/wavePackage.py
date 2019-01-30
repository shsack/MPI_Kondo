import numpy as np
from MPSModule.MatrixProducts import MPS, MPO
from qutip import sigmam, sigmap
from MPSModule.TEBD import TEBDstep, measure, measureDensityMatrix, traceMPO, truncate
import copy

def initializePure(localHilbertSpace, maximalBondDimension, coefficients, truncate = False):
    """
    Initialize a pure single particle superposition MPS with coefficients coeffs for single excitation states
    Params:
        localHilbertSpace: local Hilbert space dimension
        maximalBondDimension: maximal bond dimension
        coefficients: coefficients of wavepackage states
        truncate: wether mps should be truncated
    Returns:
        an MPS which is a superposition of single excitation states
    """
    L = len(coefficients) # number of sites is the number of given coefficients

    # initialize MPS, it has bond dimension equal to number of sites
    myMPS = MPS(**{'numberOfSites': L, 'localHilbertSpace': localHilbertSpace, 'bondDimension': L, 'maximalBondDimension': maximalBondDimension})
    myMPS.reset()

    # Fill the matrices of the MPS
    for i in range(L):
        # the singular values can be all ones
        #myMPS.S = [np.ones(myMPS.D[i]) for i in range(myMPS.L)]
        # Swap indices of matrices to get access to matrices corresponding to a site and a spin state
        myMPS.M[i] = myMPS.M[i].transpose((1,0,2))
        if i == 0:
            myMPS.M[i][0, 0, :] = np.ones((myMPS.D[i]))
            myMPS.M[i][1, 0, i] = coefficients[0]
            myMPS.M[i][0, 0, 0] = 0
        elif i == L-1:
            myMPS.M[i][0] = np.ones((myMPS.D[i]))
            myMPS.M[i][1, i, 0] = coefficients[i]
            myMPS.M[i][0, i, 0] = 0
        else:
            myMPS.M[i][0] = np.eye((myMPS.D[i-1]))
            myMPS.M[i][1, i, i] = coefficients[i]
            myMPS.M[i][0, i, i] = 0
        # swap indices back to get correct form again
        myMPS.M[i] = myMPS.M[i].transpose((1, 0, 2))

    # orthogonalize
    #if truncate:
    #    myMPS.makeCanonical(truncate = truncate)
    return myMPS

def initializeDensityMatrix(localHilbertSpace, maximalBondDimension, coefficientMatrix, truncate = False):
    """
    Initialize a density matrix in MPO form of a single excitation with coefficients from coefficientMatrix
    Params:
        localHilbertSpace: local Hilbert space dimension
        maximalBondDimension: maximal bond dimension
        coefficientMatrix: matrix of coefficients corresponding to single excitation density matrix entries
        truncate: wether MPO should be truncated
    Returns:
        an MPO that represents the desired density matrix
    """
    assert coefficientMatrix.shape[0] == coefficientMatrix.shape[1], 'Need a quadratic coefficient matrix'
    L = coefficientMatrix.shape[0]

    # initialize the density matrix as rho_00 |0><0|
    ket = MPS(**{'numberOfSites': L, 'localHilbertSpace': localHilbertSpace, 'bondDimension': L, 'maximalBondDimension': maximalBondDimension})
    bra = MPS(**{'numberOfSites': L, 'localHilbertSpace': localHilbertSpace, 'bondDimension': L, 'maximalBondDimension': maximalBondDimension})

    state = np.zeros(L, dtype = int)
    state[0] = 1

    ket.setProductState(state)
    bra.setProductState(state)
    bra = bra.conjugate()

    densityMatrix = coefficientMatrix[0][0] * (ket * bra)

    # now build the density matrix as a superposition
    for i in range(1, L):
        # find the corresponding product state of the ket for the coefficient
        ketState = np.zeros(L, dtype = int)
        ketState[i] = 1
        ket.setProductState(ketState)
        for j in range(1, L):
            # find the corresponding product state of the bra for the coefficient
            braState = np.zeros(L, dtype = int)
            braState[j] = 1
            bra.setProductState(braState)
            bra = bra.conjugate()

            densityMatrix = densityMatrix + coefficientMatrix[i][j] * (ket * bra)

    if truncate:
        densityMatrix.maxD = maximalBondDimension
        densityMatrix.makeCanonical( truncate = truncate )

    return densityMatrix

def hoppingMPO(localHilbertSpace, numberOfSites):
    """
    Create the Hopping Hamiltonian MPO
    """
    H = MPO(**{'numberOfSites': numberOfSites, 'localHilbertSpace': localHilbertSpace, 'bondDimension': 4})

    H.M[0] = np.array((np.zeros((2,2)), sigmap().full(), sigmam().full(), np.eye(2))).reshape((1,4,2,2)).transpose((0,2,3,1))
    H.M[0] = -1*H.M[0]
    H.M[-1] = np.array((np.eye(2), sigmam().full(), sigmap().full(), np.zeros((2,2)))).reshape((4,1,2,2)).transpose((0,2,3,1))
    for i in range(1, numberOfSites-1):
        H.M[i] = np.array((np.array((np.eye(2), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)))),
                           np.array((sigmam().full(), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)))),
                           np.array((sigmap().full(), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)))),
                           np.array((np.zeros((2,2)), sigmap().full(), sigmam().full(), np.eye(2))))).transpose((0,2,3,1))
    return H

def exactEnergy(waveVector, width, center, numberOfSites):
    """
    Calculate exact energy of GWP
    """
    # calculate the norm of the wavepacket
    A = np.sqrt(np.sum([np.exp(-(i - center) ** 2 / (2 * width ** 2)) for i in range(numberOfSites)]))
    # Calculate the modulation of the cosine
    B = np.sum([np.exp(-(j-center)**2/(4*width**2))*np.exp(-(j-center+1)**2/(4*width**2)) for j in range(numberOfSites-1)])

    return -2*np.cos(waveVector)*B/A**2

def planeWaveOverlapsExact(numberOfSites,  width, waveVector, center):
    """
    Calculates the exact overlaps of a gaussian wave packet with plane waves
    """
    # calculate the norm of the wavepacket
    A = np.sqrt(np.sum([np.exp(-(i-center)**2/(2*width**2)) for i in range(numberOfSites)]))

    # calculate the absolute values of the overlaps with plane waves
    overlaps = []
    k = [np.pi * (i + 1) / (numberOfSites + 1) for i in range(numberOfSites)]
    for i in range(numberOfSites):
        overlaps.append(np.sqrt(2)/(A*np.sqrt(numberOfSites+1))*(np.sum([np.exp(1j*j*waveVector)*np.exp(-(j-center)**2/(4*width**2))*np.sin(k[i]*j) for j in range(numberOfSites)])))

    return overlaps

def planeWaves(MPS):
    """
    Calculate the plane waves basis of a given system
    Input: MPS: get system parameters from this MPS
    Output: psi_k: list of plane waves as MPS in position basis
    """
    # Create the plane wave solutions to the hopping Hamiltonian as MPS
    k = [np.pi * (i + 1) / (MPS.L + 1) for i in range(MPS.L)]
    psi_k = []
    coefficients = np.zeros(MPS.L, dtype=complex)
    for i in range(MPS.L):
        for j in range(MPS.L):
            coefficients[j] = np.sqrt(2/(MPS.L+1)) * np.sin(k[i]*j)
        # coefficients /= np.linalg.norm(coefficients)
        planeWave = initializePure(MPS.d, MPS.maxD, coefficients, truncate=False)
        planeWave.normalise()
        psi_k.append(planeWave)

    # Return the plane waves
    return psi_k

def positionOperators(localHilbertSpace, numberOfSites):
    """
    Calculate the position operators for a spin chain
    Returns:
        list of position operators corresponding to single site densities
    """

    # Initialize the list of density MPOs
    operatorList = []

    # Calculate the density MPO for each site
    for site in range(numberOfSites):
        # Initialize the MPO
        operator = MPO(**{'numberOfSites': numberOfSites, 'localHilbertSpace': localHilbertSpace, 'bondDimension': 1})
        operator.reset()
        # The MPO for density at a single site consists of identities everywhere except a density operator at site
        for i in range(numberOfSites):
            if i != site:
                operator.M[i] = np.eye(operator.d[i]).reshape((1, operator.d[i], operator.d[i], 1))
            else:
                b = np.diag(np.sqrt(range(operator.d[i] - 1) + np.ones(operator.d[i] - 1)), 1)  # Annihilation
                bdag = np.diag(np.sqrt(range(operator.d[i] - 1) + np.ones(operator.d[i] - 1)), -1)  # Creation
                dens = np.dot(bdag, b)  # Density operator
                operator.M[i] = i*dens.reshape((1, operator.d[i], operator.d[i], 1))
        operatorList.append(operator)

    return operatorList

def positionSquaredOperators(localHilbertSpace, numberOfSites):
    """
    Calculate the position squared operators for a spin chain
    Returns:
        list of position squared operators corresponding to single site densities
    """

    # Initialize the list of density MPOs
    operatorList = []

    # Calculate the density MPO for each site
    for site in range(numberOfSites):
        # Initialize the MPO
        operator = MPO(**{'numberOfSites': numberOfSites, 'localHilbertSpace': localHilbertSpace, 'bondDimension': 1})
        operator.reset()
        # The MPO for density at a single site consists of identities everywhere except a density operator at site
        for i in range(numberOfSites):
            if i != site:
                operator.M[i] = np.eye(operator.d[i]).reshape((1, operator.d[i], operator.d[i], 1))
            else:
                b = np.diag(np.sqrt(range(operator.d[i] - 1) + np.ones(operator.d[i] - 1)), 1)  # Annihilation
                bdag = np.diag(np.sqrt(range(operator.d[i] - 1) + np.ones(operator.d[i] - 1)), -1)  # Creation
                dens = np.dot(bdag, b)  # Density operator
                operator.M[i] = i**2*dens.reshape((1, operator.d[i], operator.d[i], 1))
        operatorList.append(operator)

    return operatorList

def positionVariance(MP, posOps, posSquaredOps):
    """
    calculate the variance of the position of a quantum state represented as an MPS
    Params:
        MP: matrix product state or MPO representation of a quantum state (pure or as density matrix)
        posOps: position operators
        posSquaredOps: position squared operators
    Returns:
        sigma: variance of position of the quantum state
    """
    myMP = copy.deepcopy(MP)

    # measure the mean position and the mean squared position
    if type(MP) == MPS:
        meanPos = np.sum([np.abs(measure(myMP, posOps[j])) for j in range(MP.L)])
        meanSquaredPos = np.sum([np.abs(measure(myMP, posSquaredOps[j])) for j in range(MP.L)])
    elif type(MP) == MPO:
        meanPos = np.sum([np.abs(measureDensityMatrix(myMP, posOps[j])) for j in range(MP.L)])
        meanSquaredPos = np.sum([np.abs(measureDensityMatrix(myMP, posSquaredOps[j])) for j in range(MP.L)])
    return meanSquaredPos - meanPos ** 2


def meanPosition(positionOperators, numberOfSites, localHilbertSpace, maximalBondDimension, waveVector, width, center, numberOfSteps, timeEvolutionOp, timeStep):
    """
    Calculate mean position of a wave packet
    """
    if type(timeEvolutionOp) is not list:
        timeEvolutionOp = [timeEvolutionOp]*numberOfSites
    # Calculate the coefficients of a wavepackage following a Gaussian distribution
    coefficients = np.zeros(numberOfSites, dtype=complex)
    for i in range(numberOfSites):
        coefficients[i] = np.exp(-(i-center)**2/(4 * width**2))*np.exp(1j * waveVector * i)
    coefficients /= np.linalg.norm(coefficients)

    # Create the wavepackage as an MPS
    psi = initializePure(localHilbertSpace, maximalBondDimension, coefficients, truncate=False)

    # Empty array to save position
    pos = np.zeros(numberOfSteps)

    # Do time evolution and save the mean position
    for i in range(numberOfSteps):
        for j in range(0,numberOfSites-1,2):
            psi = TEBDstep(timeEvolutionOp[j], psi, j, truncate=True)
        for j in range(1,numberOfSites-1,2):
            psi = TEBDstep(timeEvolutionOp[j], psi, j, truncate=True)
        pos[i] = np.sum([np.real(measure(psi, positionOperators[l])) for l in range(numberOfSites)])

    return pos

def exactVelocity(waveVector, width, center, numberOfSites):
    """
    Calculate exact velocity of GWP
    """
    # calculate the norm of the wavepacket
    A = np.sqrt(np.sum([np.exp(-(i - center) ** 2 / (2 * width ** 2)) for i in range(numberOfSites)]))
    # Calculate the modulation of the cosine
    B = np.sum([np.exp(-(j-center)**2/(4*width**2))*np.exp(-(j-center+1)**2/(4*width**2)) for j in range(numberOfSites-1)])

    return 2*np.sin(waveVector)*B/A**2

def wavePacketCoefficients(waveVector, width, center, numberOfSites):
    """
    Calculate coefficients of a Gaussian wave packet
    """
    coefficients = np.zeros(numberOfSites, dtype=complex)
    for i in range(numberOfSites):
        coefficients[i] = np.exp(-(i-center)**2/(4*width**2))*np.exp(1j * waveVector *(i-center))
    coefficients /= np.linalg.norm(coefficients)
    return coefficients

def dephaser(densityMatrix, alpha, normalize=False):
    """
    Dephase a density matrix
    Params:
        densityMatrix: density matrix to be dephased
        alpha: dephasing parameter
    Returns:
        alpha*diag(densityMatrix)+(1-alpha)*densityMatrix
    """
    assert alpha >= 0 and alpha <= 1, 'alpha has to be between 0 and 1'

    diagDensityMatrix = copy.deepcopy(densityMatrix)
    myDensityMatrix = copy.deepcopy(densityMatrix)

    # for zero dephasing return original matrix
    if alpha==0:
        return myDensityMatrix

    diagDensityMatrix.structurePhysicalLegs()
    myDensityMatrix.structurePhysicalLegs()

    # Delete all off-diagonal elements of the matrices (only keep the sigma=sigma' entries)
    for i in range(diagDensityMatrix.L):
        for k in range(diagDensityMatrix.d[i]):
            for l in range(diagDensityMatrix.d[i]):
                if k!=l:
                    diagDensityMatrix.M[i][:,k,l,:] = np.zeros_like(diagDensityMatrix.M[i][:,k,l,:])
    # now truncate the diagonal density matrix
    diagDensityMatrix = truncate(diagDensityMatrix)
    purityDiag = np.abs(traceMPO(diagDensityMatrix*diagDensityMatrix))

    # for full dephasing return the diagonal matrix
    if alpha==1:
        return diagDensityMatrix

    # need makeScalarUnity() to store the scalar in the matrices of the MPO representation
    diagDensityMatrix = alpha * diagDensityMatrix
    diagDensityMatrix.makeScalarUnity(sites=None)
    myDensityMatrix = (1-alpha) * myDensityMatrix
    myDensityMatrix.makeScalarUnity(sites=None)
    newDensityMatrix = diagDensityMatrix + myDensityMatrix
    newDensityMatrix = truncate(newDensityMatrix)
    #newDensityMatrix.maxD = np.max(newDensityMatrix.D)
    newDensityMatrix.maxD = densityMatrix.maxD
    # truncate the new density matrix
    return newDensityMatrix

def localDephaser(densityMatrix, alpha, sites):
    """
    Dephase a density matrix at certain sites
    Params:
        densityMatrix: density matrix to be dephased
        alpha: dephasing parameter
        sites: sites where density matrix should be dephased
    Returns:
        alpha*diag(densityMatrix)+(1-alpha)*densityMatrix
    """
    assert alpha >= 0 and alpha <= 1, 'alpha has to be between 0 and 1'
    diagDensityMatrix = copy.deepcopy(densityMatrix)
    myDensityMatrix = copy.deepcopy(densityMatrix)

    diagDensityMatrix.structurePhysicalLegs()
    myDensityMatrix.structurePhysicalLegs()

    # Delete all off-diagonal elements of the matrices at sites (only keep the sigma=sigma' entries)
    for i in sites:
        for k in range(diagDensityMatrix.d[i]):
            for l in range(diagDensityMatrix.d[i]):
                if k!=l:
                    diagDensityMatrix.M[i][:,k,l,:] = np.zeros_like(diagDensityMatrix.M[i][:,k,l,:])

    # for full dephasing return the "diagonal" matrix
    if alpha==1:
        return diagDensityMatrix

    # for zero dephasing return original matrix
    elif alpha==0:
        return myDensityMatrix

    # need makeScalarUnity() to store the scalar in the matrices of the MPO representation
    diagDensityMatrix = alpha * diagDensityMatrix
    diagDensityMatrix.makeScalarUnity(sites=None)
    myDensityMatrix = (1-alpha) * myDensityMatrix
    myDensityMatrix.makeScalarUnity(sites=None)
    return diagDensityMatrix + myDensityMatrix
