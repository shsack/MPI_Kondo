import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

# Plotting style
plt.style.use('bmh')

# File to import
D = 10
file = 'simulation_results/dot_cav_D_{}.txt'.format(D)

data = pd.read_csv(file, delimiter=' ', header=None, dtype=float)
epsImp = list(data[data.columns[0]])
epsCav = list(data[data.columns[1]])

# Choose the observable as a column of the text file
num_obs = 3
observable = list(data[data.columns[num_obs]])

data = data.pivot(index=0, columns=1, values=2).T

extent = [min(epsImp), max(epsImp), min(epsCav), max(epsCav)]
plt.imshow(data, origin='lower', aspect=1.5, extent=extent)
# plt.imshow(data, origin='lower', aspect=1./1.5, extent=extent, vmin=0, vmax=2)

# Label depending of which observable is plotted
cbar = plt.colorbar()
cbar.set_label(r'$\langle n_{d\uparrow} + n_{d\downarrow} \rangle$')
plt.xlabel(r'$\epsilon_d$')
plt.ylabel(r'$\epsilon_c$')
fig = plt.gcf()
fig.savefig('plots/heatmap_D_{}.pdf'.format(D))