"""

from .localOperators import*
import numpy as np

class dotCavity(localOperators):

    #constructs the MPO of the quantum dot cavity

    def __init__(self, tL, tR, tC, tBath, omega, epsDot, epsCav, epsBath, U, h=0):

        localOperators.__init__(self)

        # nearest neighbour coupling
        self.t = list(reversed(tBath)) + [tL] + [omega] + [tC] + list(tBath)
        self.N = len(self.t) + 1

        # next nearest neighbour coupling
        self.s = [0] * (int(self.N / 2) - 1) + [tR] + [0] * (int(self.N / 2) - 1)

        # Coulomb interaction and magnetic field
        self.U = [0] * (int(self.N / 2) - 1) + [U] + [0] * int(self.N / 2)
        self.h = [0] * (int(self.N / 2) - 1) + [h] * 2 + [0] * (int(self.N / 2) - 1)

        # on site energy
        self.epsBath = list(epsBath) + [0] * 2 + list(epsBath)
        self.epsCav = [0] * int(self.N / 2) + [epsCav] + [0] * (int(self.N / 2) - 1)
        self.epsDot = [0] * (int(self.N / 2) - 1) + [epsDot] + [0] * int(self.N / 2)

        # combine on site energy
        self.on_site = [np.diag([0, self.epsBath[n] + (self.epsDot[n] - self.h[n]) + (self.epsCav[n] - self.h[n]),
                                                               self.epsBath[n] + (self.epsDot[n] + self.h[n]) + (self.epsCav[n] + self.h[n]),
                                                               2 * (self.epsBath[n] + self.epsDot[n] + self.epsCav[n]) + self.U[n]]) for n in range(self.N)]

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Hamiltonian MPO with magnetic field

        self.MPO = []

        # adding the left edge
        self.MPO.append(np.array([[self.on_site[0], self.t[0].conj() * self.create_up, self.t[0].conj() * self.create_down, self.t[0] * self.create_up, self.t[0] * self.create_down,
                                                                 self.s[0].conj() * self.annih_up, self.s[0].conj() * self.annih_down, self.s[0] * self.create_up, self.s[0] * self.create_down, self.I]]))

        # calculating the bulk up to the dot
        for n in range(1, self.N - 2):
            self.MPO.append(np.array([[self.I, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                    [self.create_up, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                    [self.create_down, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                    [self.annih_up, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                    [self.annih_down, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                    [self.Z, self.I, self.I, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                    [self.Z, self.I, self.I, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                    [self.Z, self.Z, self.Z, self.I, self.I, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                    [self.Z, self.Z, self.Z, self.I, self.I, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                    [self.on_site[n], self.t[n].conj() * self.annih_up, self.t[n].conj() * self.annih_down, self.t[n] * self.create_up, self.t[n] * self.create_down,
                                              self.s[n].conj() * self.annih_up, self.s[n].conj() * self.annih_down, self.s[n] * self.create_up, self.s[n] * self.create_down, self.I]]))

        # adding the right edge
        self.MPO.append(localOperators.localOperators.np.array([[self.I], [self.create_up], [self.create_down], [self.annih_up], [self.annih_down],
                                                                [self.Z], [self.Z], [self.Z], [self.Z], [self.on_site[-1]]]))

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Cavity occupation number MPO

        self.cavNumberMPO = []

        self.cavNumberMPO.append(localOperators.localOperators.np.array([[self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.I]]))

        for n in range(1, int(self.N / 2)):
            self.cavNumberMPO.append(localOperators.localOperators.np.array([[self.I, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.I]]))


        self.cavNumberMPO.append(localOperators.localOperators.np.array([[self.I, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                         [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                         [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                         [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                         [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                         [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                         [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                         [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                         [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                         [self.n_up + self.n_down, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.I]]))

        for n in range(0, int(self.N / 2) - 3):
            self.cavNumberMPO.append(localOperators.localOperators.np.array([[self.I, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.I]]))

        self.cavNumberMPO.append(localOperators.localOperators.np.array([[self.I], [self.Z], [self.Z], [self.Z], [self.Z], [self.Z], [self.Z], [self.Z], [self.Z], [self.Z]]))

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # entanglement between the dot and the cavity

        self.entanglement = []

        self.entanglement.append(localOperators.localOperators.np.array([[self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.I]]))

        for n in range(1, int(self.N / 2) - 1):
            self.entanglement.append(localOperators.localOperators.np.array([[self.I, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.I]]))

        self.entanglement.append(localOperators.localOperators.np.array([[self.I, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                         [self.n_up, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                         [self.n_down, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                         [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                         [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                         [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                         [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                         [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                         [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                         [self.Z, self.n_down, self.n_up, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.I]]))

        for n in range(0, int(self.N / 2) - 2):
            self.entanglement.append(localOperators.localOperators.np.array([[self.I, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                             [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.Z, self.I]]))

        self.entanglement.append(localOperators.localOperators.np.array([[self.I], [self.Z], [self.Z], [self.Z], [self.Z], [self.Z], [self.Z], [self.Z], [self.Z], [self.Z]]))


"""