from MPSModule.MatrixProducts import MPS
from MPSModule.TEBD import heisenbergEvolutionOperator, evolutionMPO, densityPlot
import numpy as np

# Set the system parameters
L = 20 # number of sites
d = 2 # local Hilbert space dimension (want a spin chain)
maxD = 10 # maximal bond dimension

# Set the Heisenberg Hamiltonian interaction strengths
# H = J/2 (S+ * S- + S- * S+) + Jz Sz * Sz
J = 1
Jz = 0 # only hopping

# Set the time step
dt = 0.05

# Get the unitary two-site evolution operator ( U = exp(-1j * dt * H) )
U = heisenbergEvolutionOperator(J, Jz, dt)

# Get the evolution MPOs for all even resp. odd sites
Uodd = evolutionMPO(U, 'odd', L, d)
Ueven = evolutionMPO(U, 'even', L, d)

# Set an initial MPS (all spins down, middle one up)
psi = MPS(**{'numberOfSites': L, 'localHilbertSpace': d, 'maximalBondDimension': maxD})
state = np.zeros(L, dtype = int)
state[int(L/2)] = 1
psi.setProductState(state)

# Now do one time step (first order Trotter decomposition: evolve even and odd sites separately)
evolvedPsi = Uodd * psi
evolvedPsi.maxD = maxD # ugly: set maximal bond dimension manually
evolvedPsi.makeCanonical(truncate = True) # Truncate the MPS (would be more efficient if it was made in the time step directly)
evolvedPsi = Ueven * evolvedPsi
evolvedPsi.maxD = maxD # ugly: set maximal bond dimension manually
evolvedPsi.makeCanonical(truncate = True) # Truncate the MPS

# evolvedPsi is now the time evolved psi for one time step!



"""
Example: density plot of a single spin hopping on the lattice
"""
N = 100 # number of time steps
densityPlot(psi, U, N)




