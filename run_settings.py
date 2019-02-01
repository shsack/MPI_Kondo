import numpy as np
from functools import partial
from itertools import product


def import_main_data(which_simulation):

    if which_simulation == 'dot_cav_purity_heatmap':

        from dot_cavity.dot_cav_full_info import main

        D = 10
        num_data_points = 16 # !!! Has to be a multiple of the requested nodes !!! <---- IMPORTANT
        epsImp = np.linspace(-1., 0.5, num_data_points)
        epsCav = np.linspace(-0.5, 0.5, num_data_points)
        data = list(product(epsImp, epsCav))
        main = partial(main, D=D)

        return data, main

    if which_simulation == 'purity_entropy_D':

        from dot.purity_entropy_D import main

        V = 0.1
        D = range(10, 70)
        main = partial(main, V=V)

        # TODO: DOES NOT YET WORK

        return D, main

    if which_simulation == 'purity_V':

        from dot.purity_V import main
        D = [20, 50, 80, 110, 140, 170, 200]
        V = np.linspace(start=0.0, stop=0.5, num=50)
        data = product(D, V)

        # TODO: DOES NOT YET WORK

        return data, main

    else:

        print('Requested simulation not found!')





