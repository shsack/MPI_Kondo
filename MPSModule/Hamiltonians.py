import numpy as np
import copy
from .MatrixProducts import MPO
from .MatrixProducts import MPS

def Heisenberg(numberOfSites, periodic):
    # TODO: make it work for arbitrary 2l+1
    operator = MPO(numberOfSites=numberOfSites,
                   bondDimension=5,
                   localHilbertSpace=2,
                   maximalBondDimension=5,
                   thresholdEntanglement=0,
                   periodic=periodic)
    assert not periodic, 'Cannot deal with periodic boundary conditions yet'

    # check these prefactors
    J = 1
    J = np.sqrt(J)

    I = np.asarray(((1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype='complex')
    X = J*np.asarray(((0, 1, 0), (1, 0, 1), (0, 1, 0)), dtype='complex') / np.sqrt(2)
    Y = J*1j * np.asarray(((0., -1, 0), (1, 0, -1), (0, 1, 0)), dtype='complex') / np.sqrt(2)
    Z = J*np.asarray(((1., 0, 0), (0, 0, 0), (0, 0, -1)), dtype='complex')

    I = np.asarray(((1, 0), (0, 1)), dtype='complex')
    X = J * np.asarray(((0, 1), (1, 0)), dtype='complex')/2
    Y = J * 1j * np.asarray(((0., -1), (1, 0)), dtype='complex')/2
    Z = J * np.asarray(((1., 0), (0, -1)), dtype='complex')/2


    zero = np.asarray(((0, 0, 0), (0, 0, 0), (0, 0, 0)), dtype='complex')

    zero = np.asarray(((0, 0), (0, 0)), dtype='complex')

    dim = 5

    tensor = np.array([[I] + [zero] * (dim - 1),
                       [X] + [zero] * (dim - 1),
                       [Y] + [zero] * (dim - 1),
                       [Z] + [zero] * (dim - 1),
                       [zero, X, Y, Z, I]])
    tensor = np.transpose(tensor, (0, 2, 3, 1))


    tensorLeft = np.array([[zero, X, Y, Z, I]])
    tensorLeft = np.transpose(tensorLeft, (0, 2, 3, 1))

    tensorRight = np.array([[I], [X], [Y], [Z], [zero]])
    tensorRight = np.transpose(tensorRight, (0, 2, 3, 1))

    operator.M = copy.deepcopy([tensorLeft] + [tensor] * (operator.L - 2) + [tensorRight])

    operator.structuredPhysicalLegs = True



    return operator

def AKLT2(numberOfSites, periodic):
    operator = MPO(numberOfSites=numberOfSites,
                   bondDimension=14,
                   localHilbertSpace=3,
                   maximalBondDimension=14,
                   thresholdEntanglement=0,
                   periodic=periodic)
    assert not periodic, 'Cannot deal with periodic boundary conditions yet'

    # check these prefactors
    J1 = 1

    J2 = np.sqrt(1 / 3)

    I = np.asarray(((1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype='complex')
    X = np.asarray(((0, 1, 0), (1, 0, 1), (0, 1, 0)), dtype='complex') / np.sqrt(2)
    Y = 1j * np.asarray(((0., -1, 0), (1, 0, -1), (0, 1, 0)), dtype='complex') / np.sqrt(2)
    Z = np.asarray(((1., 0, 0), (0, 0, 0), (0, 0, -1)), dtype='complex')

    zero = np.asarray(((0, 0, 0), (0, 0, 0), (0, 0, 0)), dtype='complex')


    XX = np.dot(X, X) * J2
    YY = np.dot(Y, Y) * J2
    ZZ = np.dot(Z, Z) * J2

    XY = np.dot(X, Y) * J2
    YZ = np.dot(Y, Z) * J2
    ZX = np.dot(Z, X) * J2
    YX = np.dot(Y, X) * J2
    ZY = np.dot(Z, Y) * J2
    XZ = np.dot(X, Z) * J2

    dim = 14

    tensor = np.array([[I] + [zero] * (dim - 1),
                      [X] + [zero] * (dim - 1),
                      [Y] + [zero] * (dim - 1),
                      [Z] + [zero] * (dim - 1),
                      [XX] + [zero] * (dim - 1),
                      [YY] + [zero] * (dim - 1),
                      [ZZ] + [zero] * (dim - 1),
                      [XY] + [zero] * (dim - 1),
                      [YZ] + [zero] * (dim - 1),
                      [ZX] + [zero] * (dim - 1),
                      [YX] + [zero] * (dim - 1),
                      [ZY] + [zero] * (dim - 1),
                      [XZ] + [zero] * (dim - 1),
                      [zero, X, Y, Z, XX, YY, ZZ, XY, YZ, ZX, YX, ZY, XZ, I]])

    tensor = np.transpose(tensor,(0,2,3,1))

    tensorLeft = np.array([[zero, X, Y, Z, XX, YY, ZZ, XY, YZ, ZX, YX, ZY, XZ, I]])
    tensorLeft = np.transpose(tensorLeft, (0, 2, 3, 1))

    tensorRight = np.array([[I], [X], [Y], [Z], [XX], [YY], [ZZ], [XY], [YZ], [ZX], [YX], [ZY], [XZ], [zero]])
    tensorRight = np.transpose(tensorRight, (0, 2, 3, 1))

    operator.M = copy.deepcopy([tensorLeft] + [tensor] * (operator.L - 2) + [tensorRight])



    operator.structuredPhysicalLegs = True



    """
    for i in range(operator.L):
        operator.M[i] = copy.deepcopy(tensor)



    if not operator.periodic:


        # should check this!!!!

        operator.M[0] = np.expand_dims(operator.M[0][-1, :, :, :], axis=0)
        operator.M[-1] = np.expand_dims(operator.M[-1][:, 0, :, :], axis = 1)

    """

    return operator

def AKLT(numberOfSites, periodic):
    operator = MPO(numberOfSites=numberOfSites,
                   bondDimension=14,
                   localHilbertSpace=3,
                   maximalBondDimension=14,
                   thresholdEntanglement=0,
                   periodic=periodic)
    assert not periodic, 'Cannot deal with periodic boundary conditions yet'

    J1 = 2

    J2 = np.sqrt(1 / 3)

    I = np.asarray(((1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype='complex')
    SX = np.asarray(((0, 1, 0), (1, 0, 1), (0, 1, 0)), dtype='complex') / np.sqrt(2)
    SY = 1j * np.asarray(((0., -1, 0), (1, 0, -1), (0, 1, 0)), dtype='complex') / np.sqrt(2)
    SZ = np.asarray(((1., 0, 0), (0, 0, 0), (0, 0, -1)), dtype='complex')

    tensor = np.zeros((14, 14, 3, 3), dtype='complex')
    tensor[1, 0] = np.einsum('ij,jk', SX, SX) * J2
    tensor[2, 0] = np.einsum('ij,jk', SY, SY) * J2
    tensor[3, 0] = np.einsum('ij,jk', SZ, SZ) * J2
    tensor[4, 0] = np.einsum('ij,jk', SX, SY) * J2
    tensor[5, 0] = np.einsum('ij,jk', SY, SZ) * J2
    tensor[6, 0] = np.einsum('ij,jk', SZ, SX) * J2
    tensor[7, 0] = np.einsum('ij,jk', SY, SX) * J2
    tensor[8, 0] = np.einsum('ij,jk', SZ, SY) * J2
    tensor[9, 0] = np.einsum('ij,jk', SX, SZ) * J2

    tensor[10, 0] = SX * J1
    tensor[11, 0] = SY * J1
    tensor[12, 0] = SZ * J1

    tensor[-1, -2] = np.einsum('ij,jk', SX, SX) * J2
    tensor[-1, -3] = np.einsum('ij,jk', SY, SY) * J2
    tensor[-1, -4] = np.einsum('ij,jk', SZ, SZ) * J2
    tensor[-1, -5] = np.einsum('ij,jk', SX, SY) * J2
    tensor[-1, -6] = np.einsum('ij,jk', SY, SZ) * J2
    tensor[-1, -7] = np.einsum('ij,jk', SZ, SX) * J2
    tensor[-1, -8] = np.einsum('ij,jk', SY, SX) * J2
    tensor[-1, -9] = np.einsum('ij,jk', SZ, SY) * J2
    tensor[-1, -10] = np.einsum('ij,jk', SX, SZ) * J2

    tensor[-1, -11] = SX * J1
    tensor[-1, -12] = SY * J1
    tensor[-1, -13] = SZ * J1

    tensor[0, 0] = I
    tensor[-1, -1] = I

    # print(tensor[0,0])
    # print(tensor[0, 5])
    # print(tensor[2, -1])
    # print(tensor[0,-1])

    tensor = tensor.transpose(0, 2, 3, 1)

    for i in range(operator.L):
        operator.M[i] = copy.deepcopy(tensor)

    if not operator.periodic:
        operator.M[0] = operator.M[0][-1, :, :, :]
        operator.M[-1] = operator.M[-1][:, :, :, 0]

    operator.structurePhysicalLegs()

    for i, matrix in enumerate(operator.M):
        a = np.sum(np.abs(matrix), axis=(1, 2))
        a = np.sum(np.sign(np.abs(a[::-1,:]-a[::-1,::].transpose())))
        #print(i)
        #print(a)

        # print(a[::-1,:]-a[::-1,::].transpose())
        #print(np.sum(np.abs(matrix), axis = (1,2))[-1,0])
        #print(matrix[-1,:,:,0])
        #print(matrix[-1,:,:,-1])

    # operator.makeCanonical('Right', truncate=True)
    # print(operator.scalar)
    # operator.makeScalarUnity(sites=None)
    # print(operator.M[4])
    return operator


def AKLTGS(numberOfSites, periodic):
    state = MPS(numberOfSites=numberOfSites,
                bondDimension=2,
                localHilbertSpace=3,
                maximalBondDimension=2,
                thresholdEntanglement=0,
                periodic=periodic)
    assert not periodic, 'Cannot deal with periodic BC yet'
    Ap = np.asarray([[0, 1], [0, 0]]) * np.sqrt(2 / 3)
    A0 = -np.asarray([[1, 0], [0, -1]]) * np.sqrt(1 / 3)
    Am = -np.asarray([[0, 0], [1, 0]]) * np.sqrt(2 / 3)

    tensor = np.asarray([Ap, A0, Am])

    tensor = tensor.transpose(1, 0, 2)

    for i in range(state.L):
        state.M[i] = tensor

    if not state.periodic:
        state.M[0] = np.expand_dims(np.sum(state.M[0], axis=0), axis=0) / np.power(2, 1 / 4)
        state.M[-1] = np.expand_dims(np.sum(state.M[-1], axis=-1), axis=-1) / np.power(2, 1 / 4)

    # for i in range(state.L):
    #    print(np.shape(state.M[i]))

    # print(state)

    state._MP__flattenInplace()

    state.normalise()

    return state
