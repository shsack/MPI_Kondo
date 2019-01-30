from mpi4py import MPI
import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import product
from dotCavity import main # Import main function

# def main(x, y):
#     return x + y

def split(rank, size, *data):

    """Split the data into chunks"""

    partitions = [int(len(d) / size) for d in data]
    return [d[rank * partition:(rank + 1) * partition] for partition, d in zip(partitions, data)]

def parallel_apply(*data):

    """Feed parameters into function."""

    p = Pool(processes=cpu_count())
    return p.starmap(main, product(*data))


# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank() # Identification number of node
size = comm.Get_size() # Number of nodes








# Define data matrix
n_x, n_y = 8, 8
data_x, data_y = list(range(n_x)), list(range(n_y))

# Split the data for the nodes and apply the function to every node
data_x, data_y = split(rank, size, data_x, data_y)
result = parallel_apply(data_x, data_y)

# Combine the results from the nodes
result = comm.gather(result, root=0)
data_x, data_y = comm.gather(data_x, root=0), comm.gather(data_y, root=0) #TODO: write gather function

# Save the results in a text file
if rank == 0:

    f = open("output_file.txt", "a")

    for res, x, y in zip(result, data_x, data_y):
        for x_y, res_i in zip(product(x, y), res):
            x, y = x_y
            f.write(str(x) + " " + str(y) + " " + str(res_i) + "\n")

    f.close()
