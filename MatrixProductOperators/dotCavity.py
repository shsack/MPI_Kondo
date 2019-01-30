from .localOperators import *
import copy


class dotCavity(localOperators):

    """constructs the MPO of the quantum dot cavity"""

    def __init__(self, tL, tR, tBathLeft, tBathRight, omega, epsDot, epsCav, epsBathLeft, epsBathRight, U, h=0):

        localOperators.__init__(self)

        # nearest neighbour coupling
        self.t = np.array(list(reversed(tBathLeft)) + [tL] + [omega] + [tR] + list(tBathRight))
        #self.N = len(self.t)

        # on site energy

        # forgot to reverse the left on site energies....
        self.epsBath = list(reversed(epsBathLeft)) + ([0] * 2) + list(epsBathRight)
        self.epsCav = [0] * len(epsBathLeft) + [0] + [epsCav] + [0] * len(epsBathRight)
        self.epsDot = [0] * len(epsBathLeft) + [epsDot] + [0] + [0] * len(epsBathRight)

        # Magnetic field for now only on dot to force the spin up
        self.h = [0] * len(epsBathLeft) + [h] * 2 + [0] * len(epsBathRight)

        # Coulomb interaction and magnetic field
        self.U = [0] * len(epsBathLeft) + [U] + [0] + [0] * len(epsBathRight)

        # Chain length
        self.N = len(self.epsBath)

        # combine on site energy
        self.on_site = [np.diag([0, self.epsBath[n] + (self.epsDot[n] - self.h[n]) + (self.epsCav[n] - self.h[n]),
                                 self.epsBath[n] + (self.epsDot[n] + self.h[n]) + (self.epsCav[n] + self.h[n]),
                                 2 * (self.epsBath[n] + self.epsDot[n] + self.epsCav[n]) + self.U[n]]) for n in range(self.N)]

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Hamiltonian MPO with magnetic field

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
                [self.on_site[n], self.t[n] * self.create_up, self.t[n] * self.create_down, self.t[n].conj() * self.annih_up,
                 self.t[n].conj() * self.annih_down, self.I]]))

        # adding the right edge
        self.Hamiltonian.append(np.array(
                [[self.I], [self.annih_up], [self.annih_down], [self.create_up], [self.create_down],
                 [self.on_site[-1]]]))

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Unity MPO

        self.unity = [np.array([[self.I]])] * self.N

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Spin up occupation at the dot

        self.n_dot_up = copy.deepcopy(self.unity)

        self.n_dot_up[int(self.N / 2) - 1] = np.array([[self.n_up]])

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Spin down occupation at the dot

        self.n_dot_down = copy.deepcopy(self.unity)

        self.n_dot_down[int(self.N / 2) - 1] = np.array([[self.n_down]])

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Spin down occupation at the cavity

        self.n_cav_down = copy.deepcopy(self.unity)

        self.n_cav_down[int(self.N / 2)] = np.array([[self.n_down]])

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Spin up occupation at the cavity

        self.n_cav_up = copy.deepcopy(self.unity)

        self.n_cav_up[int(self.N / 2)] = np.array([[self.n_up]])


        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        #  Spin up + down occupation at the dot

        self.n_dot_up_down = copy.deepcopy(self.unity)

        self.n_dot_up_down[int(self.N / 2) - 1] = np.array([[self.n_up + self.n_down]])
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Spin up + down occupation at the cavity

        self.n_cav_up_down = copy.deepcopy(self.unity)

        self.n_cav_up_down[int(self.N / 2)] = np.array([[self.n_up + self.n_down]])

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Spin up * spin down occupation at the the cavity

        self.n_dot_up_n_cav_down = [np.array([[self.I, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.I]])] * self.N

        self.n_dot_up_n_cav_down[0] = self.n_dot_up_n_cav_down[0][0, :, :, :]
        self.n_dot_up_n_cav_down[0] = self.n_dot_up_n_cav_down[0][np.newaxis, :, :, :]
        self.n_dot_up_n_cav_down[-1] = self.n_dot_up_n_cav_down[-1][:, 0, :, :]
        self.n_dot_up_n_cav_down[-1] = self.n_dot_up_n_cav_down[-1][:, np.newaxis, :, :]


        self.n_dot_up_n_cav_down[int(self.N / 2) - 1] = np.array([[self.I, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.n_up, self.Z, self.Z, self.Z, self.I]])

        self.n_dot_up_n_cav_down[int(self.N / 2)] = np.array([[self.I, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.n_down, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.I]])

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # Spin down * spin up occupation at the the cavity

        self.n_dot_down_n_cav_up = [np.array([[self.I, self.Z, self.Z, self.Z, self.Z, self.Z],
                                              [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                              [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                              [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                              [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                              [self.Z, self.Z, self.Z, self.Z, self.Z, self.I]])] * self.N

        self.n_dot_down_n_cav_up[0] = self.n_dot_down_n_cav_up[0][0, :, :, :]
        self.n_dot_down_n_cav_up[0] = self.n_dot_down_n_cav_up[0][np.newaxis, :, :, :]
        self.n_dot_down_n_cav_up[-1] = self.n_dot_down_n_cav_up[-1][:, 0, :, :]
        self.n_dot_down_n_cav_up[-1] = self.n_dot_down_n_cav_up[-1][:, np.newaxis, :, :]


        self.n_dot_down_n_cav_up[int(self.N / 2) - 1] = np.array([[self.I, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                  [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                  [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                  [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                  [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                  [self.Z, self.n_down, self.Z, self.Z, self.Z, self.I]])

        self.n_dot_down_n_cav_up[int(self.N / 2)] = np.array([[self.I, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                              [self.n_up, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                              [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                              [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                              [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                              [self.Z, self.Z, self.Z, self.Z, self.Z, self.I]])

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # still need to adjust this!!!

        # fucked up the order anyways...

        self.current = copy.deepcopy(self.unity)

        self.current[int(self.N / 2)] = np.array([[self.I, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, omega * self.create_up, omega * self.create_down,
                                                  omega * self.annih_up, omega * self.annih_down, self.I]])

        self.current[int(self.N / 2) - 1] = np.array([[self.I, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [-self.annih_up, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [-self.annih_down, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.create_up, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.create_down, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.I]])

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        self.dot_annihilation_up = copy.deepcopy(self.unity)

        self.dot_annihilation_up[int(self.N / 2) - 1] = np.array([[self.I, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                 [self.annih_up, self.Z, self.Z, self.Z, self.Z, self.I]])

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        self.cav_annihilation_up = copy.deepcopy(self.unity)

        self.cav_annihilation_up[int(self.N / 2)] = np.array([[self.I, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                  [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                  [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                  [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                  [self.Z, self.Z, self.Z, self.Z, self.Z, self.Z],
                                                                  [self.annih_up, self.Z, self.Z, self.Z, self.Z,
                                                                   self.I]])

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
