from dotCavity import dmrg as main
from mpi4py import MPI
import time
import resource

# Start the timer
start = time.time()

# MPI setup, not actually used --> just for testing with MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Identification number of node
size = comm.Get_size()  # Number of nodes

# Calling main for timing and memory usage
main(0, 0)

# Stop the timer
stop = time.time()

# Evaluate time (in sec) and memory usage (in MB)
time_for_call = stop - start
memory_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
print('Call of main took {0:.2f} seconds.'.format(time_for_call))
print('The maximum memory usage was {0:.2f} MB.'.format(memory_used))









