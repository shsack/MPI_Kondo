import numpy as np
import bisect
import copy
import numbers
# import msphys
import MPSModule
import time

length = 20

Hamiltonian = MPSModule.AKLT2(length, False)
GS = MPSModule.AKLTGS(length, periodic=False)
run = MPSModule.mpsDMRG(Hamiltonian, bondDimension=10, thresholdEntanglement=10 ** -6)



t0 = time.time()
run.groundState(3)
t1 = time.time()

print(t1 - t0)

exit()
