import numpy as np
from functools import partial
from itertools import product


def import_main_data(which_simulation):

    if which_simulation == 'dot_cav_heatmap':

        from dot_cavity.dot_cav_full_info import main

        # Define the simulation parameters
        num_data_points = 16  # !!! Has to be a multiple of the requested nodes !!! <---- IMPORTANT
        D = 10
        sweeps = 10
        Lambda = 2.0
        length = 10  # The length of the whole chain is twice this value
        U = 0.5
        omega = 0.1
        tL, tR = 0.01, 0.01
        epsImp = np.linspace(-1., 0.5, num_data_points)  # Linspace for 2D grit
        epsCav = np.linspace(-0.5, 0.5, num_data_points)
        data = list(product(epsImp, epsCav))  # Equivalent to two nested for loops over the linspaces

        # Fix the parameters except for epsImp and epsCav
        main = partial(main, D=D, U=U, omega=omega, tL=tL, tR=tR, Lambda=Lambda, sweeps=sweeps,length=length)

        return data, main

    if which_simulation == 'purity_entropy_D':

        from dot.purity_entropy_D import main

        V = 0.1
        U = 0.5
        epsImp = -U / 2
        Lambda = 2.0
        length = 20
        sweeps = 10
        D = range(10, 70)
        main = partial(main, V=V, U=U, epsImp=epsImp, Lambda=Lambda, lenght=length, sweeps=sweeps)

        # TODO: DOES NOT YET WORK

        return D, main

    if which_simulation == 'purity_V':

        from dot.purity_V import main

        U = 0.5
        epsImp = -U / 2
        Lambda = 2.0
        length = 20
        sweeps = 10
        D = [20, 50, 80, 110, 140, 170, 200]
        V = np.linspace(start=0.0, stop=0.5, num=50)

        main = partial(main, U=U, epsImp=epsImp, Lambda=Lambda, length=length, sweeps=sweeps)
        data = product(D, V)

        # TODO: DOES NOT YET WORK

        return data, main

    else:

        print('Requested simulation not found!')





