import sys
sys.path.insert(0, '..')
from MatrixProductOperators import *
from mpsMethods import *



# define number of sites and measurement points
d = 4
N = 50

# construct the log transformed couplings with arbitrary precision
# generate N-1 parameters for the bath and the mixing

c = bathCouplings(density='const', N=N, Lambda=2)


def main(epsDot, epsCav):

    # construct the couplings of the semi-infinite chain Hamiltonian
    mpo = dotCavity(tL=0.01, tR=0.01, tBathLeft=c.t, tBathRight=c.t, omega=0.1, epsDot=epsDot, epsCav=epsCav,
                              epsBathLeft=c.eps, epsBathRight=c.eps, U=0.5)

    # construct the dmrg
    dmrg = mpsDMRG(MPO=mpo.Hamiltonian, d=4, D=50)

    dmrg.groundState(sweeps=5)

    dot_1 = np.real(dmrg.expVal(mpo.n_dot_up))
    cav_1 = np.real(dmrg.expVal(mpo.n_cav_down))
    dotcav_1 = np.real(dmrg.expVal(mpo.n_dot_up_n_cav_down))

    # dot_2 = np.real(dmrg.expVal(mpo.n_dot_down))
    # cav_2 = np.real(dmrg.expVal(mpo.n_cav_up))
    # dotcav_2 = np.real(dmrg.expVal(mpo.n_dot_down_n_cav_up))

    #print("epsDot: {:} epsCav: {:} dot: {:} cav: {:} dotcav: {:}".format(epsDot, epsCav, dot, cav, dotcav))

    return dotcav_1 - dot_1*cav_1 # + dotcav_2 - dot_2*cav_2
    # return dot_1 + dot_2, cav_1 + cav_2


if __name__ == '__main__':

    epsDot = float(sys.argv[1])

    epsCav = float(sys.argv[2])

    correlation = main(epsDot, epsCav)
    # n_dot, n_cav = main(epsDot, epsCav) 

    f = open("dotCavCorrelation.txt", "a")
    f.write(str(epsCav) + " " + str(epsDot) + " " + str(correlation) + "\n")
    # f.write(str(epsCav) + " " + str(epsDot) + " " + str(n_dot) + " " + str(n_cav) +  "\n")
    f.close()




