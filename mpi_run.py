from mpi4py import MPI
from itertools import product
from dotCavity import dmrg as main
import numpy as np
# from joblib import Parallel, delayed
# import multiprocessing as mp


def split_data(size, data):

    n = len(data) // size
    return [data[i:i + n] for i in range(0, len(data), n)]



def apply_main_in_node(data):

    """Feed parameters into function."""

    # p = Pool(processes=2)
    # result = p.starmap(main, data)
    #
    # p.close()  # shut down the pool
    # p.join()
    #
    # return result


    # result = Parallel(n_jobs=4)(delayed(dmrg)(*i) for i in product(*data))
    # return result

    # jobs = []
    # for d in product(*data):
    #     p = Process(target=main, args=(*d,))
    #     jobs.append(p)
    #     p.start()
    #
    # return jobs

    result = []
    for d in data:
        result.append(main(*d))

    return result


# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank() # Identification number of node
size = comm.Get_size() # Number of nodes


# Define simulation parameters
num_data_points = 40 # !!! Has to be a multiple of the requested nodes !!! <---- IMPORTANT
epsImp = np.linspace(-1./16, 0.5/16, num_data_points)
epsCav = np.linspace(-0.5/16, 0.5/16, num_data_points)
data = list(product(epsImp, epsCav))


# Split the data in the zeroth node
if rank == 0:

    data_split = split_data(size, data)

else :

    data_split = None


# data_split = split_data(size, data)


# Scatter data from zeroth node onto other nodes and do the calculation in each node
data_in_node = comm.scatter(data_split, root=0)
result = apply_main_in_node(data_in_node)


# Combine the results from the nodes
result = comm.gather(result, root=0)


# Save the results in a text file
if rank == 0:

    f = open("output_file.txt", "w")

    myRes = []
    for res in result:
        for res_i in res:
            myRes.append(res_i)
    for dat_, res_ in zip(data, myRes):
        f.write(' '.join(map(str, dat_)) + " " + ' '.join(map(str, res_)) + '\n')

    f.close()

