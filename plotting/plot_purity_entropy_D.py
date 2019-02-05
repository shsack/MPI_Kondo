import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

# Plotting style
plt.style.use('bmh')

# File to import
simulation_name = 'purity_entropy_D'
file = '../simulation_results/{}.txt'.format(simulation_name)
data = pd.read_csv(file, delimiter=' ', header=None, dtype=float)
data = data.sort_values(by=0)
D, purity, entropy = data[0], data[1], data[2]

# Label depending of which observable is plotted
plt.plot(D, purity, linestyle='-', color='seagreen', label=r'Purity $Tr(\rho^{2})$')
plt.plot(D, entropy, linestyle='-', color='cornflowerblue', label=r'Entropy $-Tr(\rho\,\, log(\rho))$')
plt.xlabel(r'Bond dimension D')
plt.ylabel(r'Purity and Entropy')
plt.legend()
fig = plt.gcf()
fig.savefig('plots/{}.pdf'.format(simulation_name))
