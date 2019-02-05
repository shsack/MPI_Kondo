import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

# Set plotting style
plt.style.use('bmh')

# File to import
simulation_name = 'purity_V'
file = '../simulation_results/{}.txt'.format(simulation_name)
data = pd.read_csv(file, delimiter=' ', header=None, dtype=float)
data = data.groupby(0)

for D, group in data:

    group = group.sort_values(1)
    plt.plot(group[1], group[2], label='D={}'.format(int(D)))

plt.xlabel(r'V')
plt.ylabel(r'Purity $Tr(\rho^2)$')
plt.legend()
fig = plt.gcf()
fig.savefig('plots/purity_V.pdf')
