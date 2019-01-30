import numpy as np


class dot_unfolded:

    """constructs the MPO of the unfolded Anderson Impurity"""

    def __init__(self, tBath, epsImp, epsBath, U, V):

        # local operators in {/0>, /1>} basis
        self.create = np.array([[0, 1], [1, 0]])
        self.annih = np.array([[1, 0], [0, 1]])
        self.Z = np.array([[0, 0], [0, 0]])
        self.I = np.array([[1, 0], [0, 1]])
        self.n = np.array([[0, 0], [0, 1]])

        # nearest neighbour coupling
        self.t = np.array(list(reversed(tBath)) + [V] + [0] + [V] + list(tBath))

        # Coulomb interaction
        self.U = [0] * len(epsBath) + [U] + [0] * len(epsBath)

        # on site energy
        self.eps = list(reversed(epsBath)) + [epsImp] + [epsImp] + list(epsBath)

        # Chain length
        self.N = len(self.eps)

        # on site term
        self.on_site = [np.diag([0, self.eps[n]]) for n in range(self.N)]

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Hamiltonian MPO
        self.Hamiltonian = []

        # adding the left edge
        self.Hamiltonian.append(np.array([[self.on_site[0], self.U[0] * self.n, self.t[0] * self.create, self.t[0] * self.annih, self.I]]))

        # calculating the bulk
        for n in range(1, self.N-1):
            self.Hamiltonian.append(np.array([[self.I, self.Z, self.Z, self.Z, self.Z],
                                     [self.annih, self.Z, self.Z, self.Z, self.Z],
                                     [self.create, self.Z, self.Z, self.Z, self.Z],
                                    [self.n, self.Z, self.Z, self.Z, self.Z],
                [self.on_site[n], self.U[n] * self.n, self.t[n] * self.create, self.t[n] * self.annih, self.I]]))

        # adding the right edge
        self.Hamiltonian.append(np.array(
                [[self.I], [self.annih], [self.create], [self.n], [self.on_site[-1]]]))

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Occupation up MPO
        self.n_up = np.array([[[self.I]]] * self.N)
        self.n_up[int(self.N / 2)-1] = np.array([[[0, 0], [0, 1]]])

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Occupation down MPO
        self.n_down = np.array([[[self.I]]] * self.N)
        self.n_down[int(self.N / 2)] = np.array([[[0, 0], [0, 1]]])

