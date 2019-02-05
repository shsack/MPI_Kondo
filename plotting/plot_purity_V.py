import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# Set plotting style
plt.style.use('bmh')

Ds = [20, 50, 80, 110, 140, 170, 200]

for D in Ds:
    file = 'purity_V_{}.txt'.format(D)
    data = pd.read_csv(file, delimiter=' ', header=None, dtype=float)

    V = data[0]
    purity = data[1]
    purity = interp1d(x=V, y=purity)
    V = np.linspace(start=0, stop=0.5, num=100)
    plt.plot(V, purity(V), label='D={}'.format(D))

plt.xlabel(r'V')
plt.ylabel(r'Purity $Tr(\rho^2)$')
plt.legend()
fig = plt.gcf()
fig.savefig('purity_V.pdf')
