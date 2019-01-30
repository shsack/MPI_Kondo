import numpy as np
import os
import sys

for epsCav in np.linspace(-0.5/16, 0.5/16, 50):
    for epsImp in np.linspace(-1./16, 0.5/16, 50):

        command_to_launch = 'bsub -W 6:00 python dotCavCorrelationNew.py {:} {:}'.format(epsImp, epsCav)
        os.system(command_to_launch)





