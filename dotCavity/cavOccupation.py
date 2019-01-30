import sys
sys.path.insert(0, '..')
from MatrixProductOperators import *
from mpsMethods import *

# define number of sites and measurement points
d = 4
N = 50

# construct the log transformed couplings with arbitrary precision
# generate N-1 parameters for the bath and the mixing
c = bathCouplings(density='const', N=N - 1, Lambda=2)


def main(epsCav, omega):

    # construct the couplings of the semi-infinite chain Hamiltonian
    mpo = dotCavity(tL=0.01, tR=0.01, tC=0.01, tBath=c.t_analytic, omega=omega, epsDot=-0.25, epsCav=epsCav,
                            epsBath=c.eps_analytic, U=0.5, h=0)

    # construct the dmrg
    dmrg = mpsDMRG(MPO=mpo.Hamiltonian, d=4, D=50)

    # random MPS
    dmrg.randomMPS()

    # right orthomalize the state
    dmrg.rightOrthogonalize()

    dmrg.one_site_dmrg(sweeps=5, tolerance=1e-10, optimization=True)

    cavOccupation = np.real(dmrg.expVal(mpo.n_cav_up_down))

    return cavOccupation


if __name__ == '__main__':

    epsCav = float(sys.argv[1])

    omega = float(sys.argv[2])

    cavOccupation = main(epsCav, omega)

    f1 = open("cavOccupation{:}.txt".format(omega), "a")
    f1.write(str(epsCav) + " " + str(cavOccupation) + "\n")
    f1.close()




