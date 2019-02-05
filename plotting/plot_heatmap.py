import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

# Plotting style
plt.style.use('bmh')

# # File to import
# file = 'simulation_results/dot_cav_purity_heatmap.txt'
# data = pd.read_csv(file, delimiter=' ', header=None, dtype=float)
# epsImp = list(data[data.columns[0]])
# epsCav = list(data[data.columns[1]])

# Choose the observable as a column of the text file
obs_list = ['correlation', 'total_purity', 'dot_purity', 'cavity_purity',
            'total_occupation', 'dot_occupation', 'cavity_occupation']

for i, observable in enumerate(obs_list):

    # File to import
    file = '../simulation_results/dot_cav_heatmap.txt'
    data = pd.read_csv(file, delimiter=' ', header=None, dtype=float)
    epsImp = list(data[data.columns[0]])
    epsCav = list(data[data.columns[1]])

    num_obs = 2 + i

    data = data.pivot(index=0, columns=1, values=num_obs)
    extent = [min(epsImp), max(epsImp), min(epsCav), max(epsCav)]
    plt.imshow(data, origin='lower', aspect=1.5, extent=extent)

    # Label depending of which observable is plotted
    cbar = plt.colorbar()
    cbar.set_label(r'{}'.format(observable.replace('_', ' ')))
    plt.xlabel(r'$\epsilon_c$')
    plt.ylabel(r'$\epsilon_d$')
    fig = plt.gcf()
    fig.savefig('plots/dot_cav_heatmap_{}.pdf'.format(observable))
    plt.close()
