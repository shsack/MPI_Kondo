from mpi4py import MPI
import time
import resource
import sys
from run_settings import import_main_data
simulation_name = 'dot_cav_purity_heatmap'
_, main = import_main_data(which_simulation=simulation_name)

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

# Evaluate time (in sec) and memory usage (in MB) in rank 0
if rank == 0:
    time_for_call = stop - start
    memory_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    print('Call of main took {0:.2f} minutes.'.format(time_for_call / 60))
    print('The maximum memory usage was {0:.2f} MB.'.format(memory_used))









