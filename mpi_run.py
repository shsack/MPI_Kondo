from mpi4py import MPI
import multiprocessing as mp
from run_settings import import_main_data

# Choose the simulation you want to run
simulation_list = ['dot_cav_heatmap', 'purity_entropy_D', 'purity_V']
simulation_name = simulation_list[0]
data, main = import_main_data(which_simulation=simulation_name)


def split_data(size, data):

    n = len(data) // size
    return [data[i:i + n] for i in range(0, len(data), n)]


def apply_main_in_node(data):

    return [main(*d) for d in data]
    # p = mp.Pool(mp.cpu_count()) # TODO: fix number of processes
    # return p.starmap(main, data) # FIXME: works only for specific numpy BLAS library


# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Identification number of node
size = comm.Get_size()  # Number of nodes


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

    f = open("simulation_results/{}.txt".format(simulation_name), "w")

    myRes = []
    for res in result:
        for res_i in res:
            myRes.append(res_i)
    if simulation_name == simulation_list[0]:
        for dat_, res_ in zip(data, myRes):
            f.write(' '.join(map(str, dat_)) + " " + ' '.join(map(str, res_)) + '\n')
    else:
        for dat_, res_ in zip(data, myRes):
            f.write(str(dat_) + " " + str(res_) + '\n')

    f.close()

