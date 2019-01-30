import copy
from .localOperators import *


class dotTwoLeads(localOperators):

    def __init__(self, epsBathLeft, epsBathRight, tBathLeft, tBathRight, tL, tR, U, epsDot):

        localOperators.__init__(self)

        # is the np.array actually needed? -> check this

        self.t = list(reversed(tBathLeft)) + [tL] + [tR] + list(tBathRight)

        # watch out for the length
        self.eps = list(reversed(epsBathLeft)) + [epsDot] + list(epsBathRight)

        self.N = len(self.eps)

        self.U = [0] * len(epsBathLeft) + [U] + [0] * len(epsBathRight)

        self.dot_pos = len(epsBathLeft)

        self.on_site = [np.diag([0, self.eps[n], self.eps[n], 2 * self.eps[n] + self.U[n]]) for n in range(self.N)]

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Hamiltonian MPO of the two lead quantum dot without magnetic field

        self.Hamiltonian = []

        # adding the left edge
        self.Hamiltonian.append(np.array([[self.on_site[0], self.t[0] * self.create_up, self.t[0] * self.create_down, self.t[0].conj() * self.annih_up,
                   self.t[0].conj() * self.annih_down, self.I]]))

        # calculating the bulk
        for n in range(1, self.N-1):
            self.Hamiltonian.append(np.array([[self.I, self.Z, self.Z, self.Z, self.Z, self.Z],
                                     [self.annih_up, self.Z, self.Z, self.Z, self.Z, self.Z],
                                     [self.annih_down, self.Z, self.Z, self.Z, self.Z, self.Z],
                                     [self.create_up, self.Z, self.Z, self.Z, self.Z, self.Z],
                                     [self.create_down, self.Z, self.Z, self.Z, self.Z, self.Z],
                [self.on_site[n], self.t[n] * self.create_up, self.t[n] * self.create_down, self.t[n] * self.annih_up,
                 self.t[n] * self.annih_down, self.I]]))

        # adding the right edge
        self.Hamiltonian.append(np.array(
                [[self.I], [self.annih_up], [self.annih_down], [self.create_up], [self.create_down],
                 [self.on_site[-1]]]))

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Unity MPO

        self.zero = []

        self.zero.append(np.array([[self.Z, self.Z, self.Z, self.Z, self.Z, self.I]]))

        for n in range(1, self.N - 1):

            self.zero.append(np.array([[self.I, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.I]]))

        self.zero.append(np.array([[self.I], [self.Z], [self.Z], [self.Z], [self.Z], [self.Z]]))

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Current MPO

        # need to add the proper tunneling amplitudes....

        self.current = copy.deepcopy(self.zero)

        self.current[self.dot_pos - 1] = np.array([[self.I, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.annih_up, self.annih_down, self.create_up, self.create_down, self.I]])

        self.current[self.dot_pos] = np.array([[self.I, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.create_up, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.create_down, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [-self.annih_up, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [-self.annih_down, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.annih_up, self.annih_down, self.create_up, self.create_down, self.I]])

        self.current[self.dot_pos + 1] = np.array([[self.I, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.create_up, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.create_down, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [-self.annih_up, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [-self.annih_down, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.I]])
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Occupation number MPO

    def occupationAtSite(self, position):

        self.occupation = copy.deepcopy(self.zero)

        self.occupation[position] = np.array([[self.I, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.n_up + self.n_down, self.Z, self.Z, self.Z, self.Z, self.I]])

        if position == 0:

            self.occupation[0] = np.array([[self.n_up + self.n_down, self.Z, self.Z, self.Z, self.Z, self.I]])

        if position == self.N:

            self.occupation[self.N] = np.array([[self.I], [self.Z], [self.Z], [self.Z], [self.Z], [self.n_up + self.n_down]])

        return self.occupation










