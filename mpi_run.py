from mpi4py import MPI
# from multiprocessing import Pool, cpu_count
from itertools import product
from dotCavity import dmrg as main
import numpy as np
# from joblib import Parallel, delayed

def split(rank, size, *data):

    """Split the data into chunks"""

    partitions = [int(len(d) / size) for d in data]
    return [d[rank * partition:(rank + 1) * partition] for partition, d in zip(partitions, data)]


def parallel_apply(data):

    """Feed parameters into function."""

    #TODO: put for loop!

    # p = Pool(processes=1)
    # result = p.starmap(dmrg, product(*data))
    # p.close()  # shut down the pool
    # p.join()

    result = []
    for d in product(*data):
        result.append(main(*d))

    return result

    # result = Parallel(n_jobs=4)(delayed(dmrg)(*i) for i in product(*data))
    # return result


# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank() # Identification number of node
size = comm.Get_size() # Number of nodes


# Define simulation parameters
epsImp = np.linspace(-1./16, 0.5/16, 50)
epsCav = np.linspace(-0.5/16, 0.5/16, 50)


# Split the data for the nodes and apply the function to every node
data = split(rank, size, epsImp, epsCav)
result = parallel_apply(data)


# Combine the results from the nodes
result = comm.gather(result, root=0)
data = comm.gather(data, root=0)

# Save the results in a text file
if rank == 0:

    f = open("output_file.txt", "w")

    for res, dat in zip(result, data):
        for res_i, dat_i in zip(res, product(*dat)):
            f.write(' '.join(map(str, dat_i)) + " " + ' '.join(map(str, res_i)) + '\n')

    f.close()
