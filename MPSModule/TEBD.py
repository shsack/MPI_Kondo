import numpy as np
from _.MatrixProducts import MPO, MPS
from qutip import sigmap, sigmam, sigmaz
from scipy.linalg import expm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import copy


def evolutionMPO(evolutionOperator, oddity, numberOfSites, localHilbertSpace):
    # TODO docstring
    # TODO use underlying indexOrderInplace
    """

    :param evolutionOperator:
    :param oddity:
    :param numberOfSites:
    :param localHilbertSpace:
    :return:
    """
    """
    Get the evolution MPO corresponding to connection of two-site evolutionOperator on all even / odd sites
    Up to now: only valid for open boundary conditions (I think...)
    Params:
        evolutionOperator: local two-site evolution operator
        oddity: odd -> odd sites evolution MPO
                even -> even sites evolution MPO
        numberOfSites: number of Sites of system
        localHilbertSpace: local Hilbert space dimensions of the sites
    """

    assert oddity in ['even', 'odd'], 'No valid site configuration'
    assert len(evolutionOperator.shape) == 4, 'evolutionOperator is not a two-site operator'
    assert evolutionOperator.shape[0] == evolutionOperator.shape[2] and evolutionOperator.shape[1] == evolutionOperator.shape[3], \
        'Local Hilbert space dimensions do not match'

    # Reshape the evolutionOperator to construct MPO's (d1, d2, d1', d2') -> (d1*d1', d2*d2')
    d1 = evolutionOperator.shape[0] # Hilbert space dimension of first site
    d2 = evolutionOperator.shape[-1] # Hilbert space dimension of second site

    evolutionOperator = evolutionOperator.transpose((0, 2, 1, 3)).reshape((int(d1 ** 2), int(d2 ** 2)))

    # do a SVD and a reshape to get the MPOs
    u, s, vh = np.linalg.svd(evolutionOperator, full_matrices=False)

    # discard zero singular values
    s = np.trim_zeros(s)

    # Matrix bond dimension is given by number of singular values
    # Reshape the UL and UR into MPOs:
    #   UL: ((d1,d1'),k) -> (1,d1,d1',k)
    #   UR: (k,(d2,d2')) -> (k,d2,d2',1)
    bondDim = len(s)
    UL = np.dot(u, np.diag(np.sqrt(s))).reshape((d1, d1, 1, bondDim)).transpose((2, 0, 1, 3))
    UR = np.dot(np.diag(np.sqrt(s)), vh).reshape((bondDim, 1, d2, d2)).transpose((0, 2, 3, 1))

    # Set the bond dimensions of the evolution MPO (following Schollwoeck p77f)
    # Odd bonds evolution, even number of sites: MPO = UL UR UL UR ... UL UR
    # Odd bonds evolution, odd number of sites: MPO = UL UR UL UR ... UL UR I
    # Even bonds evolution, even number of sites: MPO = I UL UR ... UL UR I
    # Even bonds evolution, odd number of sites: MPO = I UL UR ... UL UR
    if oddity == 'odd':
        bondDimension = [bondDim, 1] * int(numberOfSites / 2)
        if numberOfSites % 2 == 1:
            bondDimension.append(1)
    else:
        bondDimension = [bondDim, 1] * int((numberOfSites - 1) / 2)
        if numberOfSites % 2 == 0:
            bondDimension.append(1)
        bondDimension = [1] + bondDimension

    # Construct an empty MPO to store the evolution matrices, set its bond dimensions
    evolutionOperator = MPO(**{'numberOfSites': numberOfSites, 'localHilbertSpace': localHilbertSpace, 'bondDimension': bondDim})
    evolutionOperator.D = bondDimension

    # Set the matrices of the MPO according to the bond dimensions:
    # D[i-1] = D[i] = 1 -> identity
    # D[i-1] = 1, D[i] = k -> UL
    # D[i-1] = k, D[i] = 1 -> UR
    for i in range(numberOfSites):
        if evolutionOperator.D[i-1] == 1:
            if evolutionOperator.D[i] == 1:
                evolutionOperator.M[i] = np.eye(evolutionOperator.d[i]).reshape((1, evolutionOperator.d[i], evolutionOperator.d[i], 1))
            else:
                evolutionOperator.M[i] = UL
        else:
            evolutionOperator.M[i] = UR

    return evolutionOperator

def getEntEntropy(MPS):
    # TODO docstring
    """

    :param MPS:
    :return:
    """
    """
    Get entanglement entropy of each bond for given MPS
    """
    E = []
    for i in range(MPS.L):
        x = MPS.S[i]**2
        E.append(-np.inner(x, np.log(x)))
    return E

def measure(MPS, MPO):
    # TODO docstring
    # TODO use built in functions of classes ( MPS.measure() )
    """

    :param MPS:
    :param MPO:
    :return:
    """
    """
    Measure an MPO for an MPS
    Returns:
        <MPS|MPO|MPS> / <MPS|MPS>
    """
    bra = MPS.conjugate()
    norm = MPS.norm()

    return (bra * (MPO * MPS)) / norm

def measureDensityMatrix(densityMatrix, observable):
    # TODO docstring
    # TODO create built in MPO function ( MPO.trace() )
    """

    :param densityMatrix:
    :param observable:
    :return:
    """
    """
    Measure an observable in a system described by a density matrix
    Params:
        densityMatrix: density matrix of the system
        observable: observable to measure
    returns: trace(ovservable*densityMatrix)/trace(densityMatrix)
    """
    return traceMPO(observable*densityMatrix)/traceMPO(densityMatrix)

def densityOperators(localHilbertSpace, numberOfSites):
    # TODO docstring
    # TODO move to appropriate file (constructors ?)
    """

    :param localHilbertSpace:
    :param numberOfSites:
    :return:
    """
    """
    Calculate the density operators for a spin chain
    Returns:
        list of density operators corresponding to single site densities
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
                operator.M[i] = dens.reshape((1, operator.d[i], operator.d[i], 1))
        operatorList.append(operator)

    return operatorList

def heisenbergEvolutionOperator(J, Jz, dt):
    # TODO docstring
    # TODO move to appropriate file (constructors ?)
    """

    :param J:
    :param Jz:
    :param dt:
    :return:
    """
    """
    Constructs the two-site imaginary or real time Evolution operator in MPO shape for a Heisenberg Hamiltonian
    Params:
        J: XY coupling strength
        Jz: spin coupling strength
        dt: time step (real or imaginary)
    Returns:
        U = exp(-1j*dt*H)
    """

    # Construct the Hamiltonian
    H = J/2 * (np.kron(sigmap().full(), sigmam().full()) + np.kron(sigmam().full(), sigmap().full())) \
        + 0.5**2 * Jz * np.kron(sigmaz().full(), sigmaz().full())

    return expm(-1j * dt * H).reshape((2, 2, 2, 2))

def densityPlot(MPS, evolutionOperator, numberOfSteps):
    # TODO docstring
    """

    :param MPS:
    :param evolutionOperator:
    :param numberOfSteps:
    :return:
    """
    """
    creates a density plot for the time evolution of an MPS for a given number of time steps
    Params:
        MPS: matrix product state
        evolutionOperator: two-site evolution operator for the system
        numberOfSteps: number of time steps to be performed
    """

    myMPS = copy.deepcopy(MPS)
    maxD = myMPS.maxD # store maximal bond dimension to truncate in evolution

    # Get the evolution operators for even and odd bonds:
    Uodd = evolutionMPO(evolutionOperator, 'odd', MPS.L, MPS.d)
    Ueven = evolutionMPO(evolutionOperator, 'even', MPS.L, MPS.d)

    # Create a plot for the density
    #plt.ion()
    fig, ax = plt.subplots()
    fig.show()

    # density array to store the densities during evolution
    densities = np.zeros((numberOfSteps, MPS.L))

    # List of density operators
    densityList = densityOperators(myMPS.d, myMPS.L)

    # Do the time evolution (odd and even bonds separate) with truncations
    for i in range(numberOfSteps):
        myMPS = Uodd * myMPS
        myMPS.maxD = maxD
        myMPS.makeCanonical(truncate=True)
        myMPS = Ueven * myMPS
        myMPS.maxD = maxD
        myMPS.makeCanonical(truncate=True)

        densities[i] = [np.abs(measure(myMPS, densityList[j])) for j in range(myMPS.L)]
        #densities[i] = [myMPS.conjugate()*densityList[j]*myMPS for j in range(myMPS.L)]

        ax.imshow(densities)
        plt.draw()

    plt.xlabel('Sites')
    plt.ylabel('time (a.u.)')
    plt.show()

def traceMPO(MPO):
    # TODO docstring
    # TODO make class method of MPO
    """

    :param MPO:
    :return:
    """
    """
    Take the trace of an MPO (contract physical indices of each site, multiply resulting matrices)
    """
    L = MPO.L

    operator = copy.deepcopy(MPO)
    operator.structurePhysicalLegs()

    matrices = operator.M[0]
    shape = matrices.shape
    matrices = matrices.reshape((shape[0], operator.d[0], operator.d[0], shape[-1]))
    temp = np.trace(matrices, axis1=1, axis2=2)

    for i in range(1, L):
        matrices = operator.M[i]
        shape = matrices.shape
        matrices = matrices.reshape((shape[0], operator.d[i], operator.d[i], shape[-1]))
        # there is a faster contraction?! first "dot", then trace...
        temp = np.dot(temp, np.trace(matrices, axis1 = 1, axis2 = 2))

    # If the system has periodic boundary conditions, we need to take the trace
    if operator.periodic:
        return np.trace(temp)
    else:
        return temp[0,0]

def TEBDstep(evolutionOperator, MPS, site, truncate = False, threshold=1e-10):
    # TODO docstring
    """

    :param evolutionOperator:
    :param MPS:
    :param site:
    :param truncate:
    :param threshold:
    :return:
    """
    """
    Do one TEBD step (apply evolutionOperator on bond betw. site and site+1 of the MPS)
    We follow the guidelines of Schollwoeck, p84f with Hastings improvement
    Params:
        evolutionOperator: two-site evolution operator
        MPS: matrix product state
        site: first site of bond to be evolved
        truncate: wether the bond dimensions should be truncated after applying the operator
    Returns:
        evolved MPS
    """

    site = int(site)
    assert len(evolutionOperator.shape) == 4, 'Not a two-site operator'
    assert evolutionOperator.shape[0] == evolutionOperator.shape[2] and evolutionOperator.shape[1] == \
           evolutionOperator.shape[3], \
        'Local Hilbert space dimensions do not match'
    assert evolutionOperator.shape[0] == MPS.d[site] and evolutionOperator.shape[-1] == MPS.d[(site + 1)%MPS.L], \
        'Hilbert space dimensions of MPS and evolution operator do not match'

    myMPS = copy.deepcopy(MPS)

    # build the psibar tensor
    psiBar = np.einsum('ijk,klm->ijlm', myMPS.M[site], myMPS.M[(site+1)%myMPS.L])

    # Apply the time evolution operator
    phiBar = np.einsum('abjl,ijlm->iabm', evolutionOperator, psiBar)

    # we get phi by multiplying with the singular values (on first site just do nothing)
    # Is this correct for the first site in our convention?
    if site == 0:
        phi = phiBar
    else:
        phi = np.einsum('i,iabm->iabm', myMPS.S[site-1], phiBar)


    # Reshape phi and do an SVD
    a1 = phi.shape[0]
    d1 = phi.shape[1]
    d2 = phi.shape[2]
    a2 = phi.shape[3]
    phi = phi.reshape((a1*d1, d2*a2))

    u, s, vh = np.linalg.svd(phi, full_matrices=False)

    # Discard negligible singular values
    s = s[np.where(s > threshold)]

    if truncate:
        s = s[:np.min([myMPS.maxD, len(s)])]

    # Restore the new matrices, set new bond dimension
    myMPS.S[site] = s
    myMPS.M[(site+1)%myMPS.L] = vh[:len(s),:].reshape((len(s), d2, a2))
    myMPS.D[site] = len(s)
    myMPS.M[site] = np.einsum('iabm,kbm->iak', phiBar, myMPS.M[(site+1)%myMPS.L].conjugate())

    return myMPS

def TEBDStepDensityMatrix(evolutionOperator, densityMatrix, site, truncate=False, threshold=1e-10, trackError=False, normalize=False):
    # TODO docstring
    """

    :param evolutionOperator:
    :param densityMatrix:
    :param site:
    :param truncate:
    :param threshold:
    :param trackError:
    :return:
    """
    """
    Do a TEBD step at bond between site and site+1 of a density matrix
    Params:
        evolutionOperator: time evolution operator
        densityMatrix: density matrix to evolve
        site: first site of bond to be evolved
        truncate: wether bond dimension should be truncated after applying the operator
        threshold: maximal value of kept singular values
    Returns:
        evolved density Matrix
    """

    site = int(site)
    assert site >= 0 and site < densityMatrix.L, 'Not a valid site'
    assert len(evolutionOperator.shape) == 4, 'Not a two-site operator'
    assert evolutionOperator.shape[0] == evolutionOperator.shape[2] and evolutionOperator.shape[1] == \
           evolutionOperator.shape[3], \
        'Local Hilbert space dimensions do not match'
    assert evolutionOperator.shape[0] == densityMatrix.d[site] and evolutionOperator.shape[-1] == densityMatrix.d[(site + 1) % densityMatrix.L], \
        'Hilbert space dimensions of MPS and evolution operator do not match'

    myDensityMatrix = copy.deepcopy(densityMatrix)
    myDensityMatrix.structurePhysicalLegs()

    # get the purity if should be normalized
    if normalize:
        sqrtPurity = np.linalg.norm(myDensityMatrix.S[site])

    # contract matrix bonds at site and site+1
    phiBar = np.tensordot(myDensityMatrix.M[site], myDensityMatrix.M[(site+1)%myDensityMatrix.L], axes=(3,0)).transpose((0,1,3,2,4,5))

    # apply the evolution operator
    phiBar = np.tensordot(evolutionOperator, phiBar, ([2,3],[1,2])).transpose((2,0,1,3,4,5))
    phiBar = np.tensordot(evolutionOperator.conjugate(), phiBar, ([0,1],[3,4])).transpose((2,3,4,0,1,5))

    # Get phi by multiplying with singular values
    #phi = np.tensordot(myDensityMatrix.S[site-1], phiBar, 0)
    if site == 0:
        phi = phiBar
    else:
        phi = np.einsum('i,iabcde->iabcde', myDensityMatrix.S[site-1], phiBar)

    # Reshape phi and do an SVD
    a1, d1, d2, d1prime, d2prime, a2 = phi.shape
    phi = phi.transpose((0,1,3,2,4,5))
    phi = phi.reshape((a1*d1*d1prime, d2*d2prime*a2))

    u, s, vh = np.linalg.svd(phi, full_matrices=False)

    # Discard negligible singular values
    strunc = s[np.where(s > threshold)]

    if truncate:
        strunc = strunc[:np.min([myDensityMatrix.maxD, len(s)])]

    # normalize the singular values to the purity
    if normalize:
        strunc *= sqrtPurity/np.linalg.norm(strunc)

    # track error
    error = np.sum(s[len(strunc):]**2)

    # Restore new matrices, set new bond dimension
    myDensityMatrix.S[site] = strunc
    myDensityMatrix.M[(site+1)%myDensityMatrix.L] = vh[:len(strunc),:].reshape((len(strunc), d2, d2prime, a2))
    myDensityMatrix.D[site] = len(strunc)
    myDensityMatrix.M[site] = np.tensordot(phiBar, myDensityMatrix.M[(site+1)%myDensityMatrix.L].conjugate(), ([5,2,4],[3,1,2]))

    if trackError:
        return myDensityMatrix, error

    return myDensityMatrix

def heisenbergEvolutionOperatorOnSite(J, Jz, dt, U1, U2):
    # TODO docstring
    """

    :param J:
    :param Jz:
    :param dt:
    :param U1:
    :param U2:
    :return:
    """
    """
    Constructs the two-site imaginary or real time Evolution operator in MPO shape for a Heisenberg Hamiltonian
    Params:
        J: XY coupling strength
        Jz: spin coupling strength
        dt: time step (real or imaginary)
        U1: on site energy on site 1
        U2: on site energy on site 2
    Returns:
        U = exp(-1j*dt*H)
    """

    # Construct the Hamiltonian
    H = J/2 * (np.kron(sigmap().full(), sigmam().full()) + np.kron(sigmam().full(), sigmap().full())) \
        + 0.5**2 * Jz * np.kron(sigmaz().full(), sigmaz().full()) \
        + U1 * np.kron(np.dot(sigmam().full(), sigmap().full()), np.eye(2)) \
        + U2 * np.kron(np.eye(2), np.dot(sigmam().full(), sigmap().full()))

    return expm(-1j * dt * H).reshape((2, 2, 2, 2))

def parcourEvolutionOperators(J, Jz, dt, onSiteEnergies):
    # TODO docstring
    """

    :param J:
    :param Jz:
    :param dt:
    :param onSiteEnergies:
    :return:
    """
    """
    create the time evolution operators for arbitrary on site energies
    Params:
        J: XY coupling strength
        Jz: spin coupling strength
        dt: time step (real or imaginary)
        onSiteEnergies: list of on site energies
    """
    operators = []
    for i in range(len(onSiteEnergies)-1):
        operators.append(heisenbergEvolutionOperatorOnSite(J, Jz, dt, onSiteEnergies[i], onSiteEnergies[i+1]))
    return operators

def getCoefficient(MPS, state):
    # TODO docstring
    """

    :param MPS:
    :param state:
    :return:
    """
    """
    get the coefficient of the MPS corresponding to state
    """
    M = copy.deepcopy(MPS.M[0][:,state[0],:])
    for i in range(1, len(state)):
        M = np.dot(M, MPS.M[i][:, state[i],:])
    return M

def truncate(MP):
    # TODO docstring
    # TODO include into the the class MP as compress
    """

    :param MP:
    :return:
    """
    """
    truncate an MP until bond dimension converges
    """
    currentD = np.zeros(MP.L)

    while not (currentD == MP.D).all():
        currentD = MP.D
        if type(MP) == MPS:
            for j in range(0, MP.L -1):
                MP = TEBDstep(np.kron(np.eye(2), np.eye(2)).reshape((2, 2, 2, 2)), MP, j, truncate=True)
            # for j in range(1, MP.L -1, 2):
            #     MP = TEBDstep(np.kron(np.eye(2), np.eye(2)).reshape((2, 2, 2, 2)), MP, j, truncate=True)
        if type(MP) == MPO:
            for j in range(0, MP.L -1):
                MP = TEBDStepDensityMatrix(np.kron(np.eye(2), np.eye(2)).reshape((2, 2, 2, 2)), MP, j, truncate=True)
            # for j in range(1, MP.L -1, 2):
            #     print('step: ', j)
            #     MP = TEBDStepDensityMatrix(np.kron(np.eye(2), np.eye(2)).reshape((2, 2, 2, 2)), MP, j, truncate=True)
    return MP

def continousOrder(array, order='linear'):
    """
    Order a 2D array such that the rows are the most continuous graphs
    """
    orderedArray = copy.deepcopy(array)
    assert len(orderedArray.shape)==2, 'not a 2D array'
    numRows, numCols = orderedArray.shape

    # function for quadratic fit
    f = lambda x, a, b, c: a*x**2 + b*x + c

    # first do linear fit
    for i in range(2, 4):
        for j in range(numRows):
            xNext = 2 * orderedArray[j, i - 1] - orderedArray[j, i - 2]
            index = (np.abs(orderedArray[j:, i] - xNext)).argmin()+j
            temp = orderedArray[j, i]
            orderedArray[j, i] = orderedArray[index, i]
            orderedArray[index, i] = temp

    # now choose wether fit should be quadratic or linear
    for i in range(4, numCols):
        for j in range(numRows):
            if order=='quadratic':
                popt, pcov = curve_fit(f, np.array(range(4)), orderedArray[j, i-4:i])
                aopt, bopt, copt = popt
                xNext = f(4, aopt, bopt, copt)
            elif order=='linear':
                xNext = 2 * orderedArray[j, i - 1] - orderedArray[j, i - 2]
            else:
                raise ValueError('Not a legitimate order')
            index = (np.abs(orderedArray[j:, i] - xNext)).argmin()+j
            temp = orderedArray[j, i]
            orderedArray[j, i] = orderedArray[index, i]
            orderedArray[index, i] = temp
    return orderedArray







