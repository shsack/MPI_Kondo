import numpy as np
from functools import partial
from itertools import product


def import_main_data(which_simulation):

    if which_simulation == 'dot_cav_heatmap':

        from dot_cavity.dot_cav_full_info import main

        # Define the simulation parameters
        num_data_points = 72  # !!! Has to be a multiple of the requested nodes !!! <---- IMPORTANT
        D = 40
        sweeps = 5
        Lambda = 2.0
        length = 20  # The length of the whole chain is twice this value
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

        # Define the simulation parameters
        V = 0.1
        U = 0.5
        epsImp = -U / 2
        Lambda = 2.0
        length = 20
        sweeps = 5
        D = list(range(10, 71))

        # Fix parameters
        data = [(d, ) for d in D]
        main = partial(main, V=V, U=U, epsImp=epsImp, Lambda=Lambda, length=length, sweeps=sweeps)

        return data, main

    if which_simulation == 'purity_V':

        from dot.purity_V import main

        # Define the simulation parameters
        U = 0.5
        epsImp = -U / 2
        Lambda = 2.0
        length = 10
        sweeps = 5
        D = [20, 50, 80, 110, 140, 170, 200]
        V = np.linspace(start=0.0, stop=0.5, num=30)

        # Fix parameters
        main = partial(main, U=U, epsImp=epsImp, Lambda=Lambda, length=length, sweeps=sweeps)
        data = list(product(D, V))

        return data, main

    else:

        print('Requested simulation not found!')





