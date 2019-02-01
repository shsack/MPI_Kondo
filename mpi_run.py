from mpi4py import MPI
from itertools import product
from dot_cavity import dmrg as main
import numpy as np
from functools import partial


def split_data(size, data):

    n = len(data) // size
    return [data[i:i + n] for i in range(0, len(data), n)]


def apply_main_in_node(data):

    return [main(*d) for d in data]


# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Identification number of node
size = comm.Get_size()  # Number of nodes

# Define simulation parameters
num_data_points = 4  # !!! Has to be a multiple of the requested nodes !!! <---- IMPORTANT
epsImp = np.linspace(-1./16, 0.5/16, num_data_points)
epsCav = np.linspace(-0.5/16, 0.5/16, num_data_points)
data = list(product(epsImp, epsCav))
D = 10
main = partial(main, D=D)

# Split the data in the zeroth node
if rank == 0:
    data_split = split_data(size, data)
else:
    data_split = None

# Scatter data from zeroth node onto other nodes and do the calculation in each node
data_in_node = comm.scatter(data_split, root=0)
result = apply_main_in_node(data_in_node)

# Combine the results from the nodes
result = comm.gather(result, root=0)

# Save the results in a text file
if rank == 0:

    f = open("simulation_results/dot_cav_D_{}.txt".format(D), "w")

    myRes = []
    for res in result:
        for res_i in res:
            myRes.append(res_i)
    for dat_, res_ in zip(data, myRes):
        f.write(' '.join(map(str, dat_)) + " " + ' '.join(map(str, res_)) + '\n')

    f.close()

