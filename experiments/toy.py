import numpy as np
import matplotlib.pyplot as plt
from visualization.visualize import plot

# [Figure 3.3 in the thesis] Run this script to generate the plot!

def rho_exact(s, alpha, beta, L):
    num = np.sinh(beta*np.sqrt(s))*np.sinh((L-alpha)*np.sqrt(s))
    denum = np.sinh(alpha*np.sqrt(s))*np.sinh((L-beta)*np.sqrt(s))
    return num / denum

def rho_bound(s, alpha, beta):
    return np.exp(-2*(alpha - beta)*np.real(np.sqrt(s)))

# Define the parameters for the test
L = 1
s = np.linspace(0.01, 20, 1000)
test = [(0.55, 0.45), (0.6, 0.4), (0.7, 0.3)]

# Plot the results
colors = ["#045B8E", "#ED2939", "#3F9C35"]
figure, ax = plt.subplots(dpi = 150)
for i, (alpha, beta) in enumerate(test):
    plot(s, rho_bound(s, alpha, beta), ax = ax, xlabel = r"$s$", ylabel = r"$|\rho(s)|$", logscale = False, xlim = (0, 20), ylim = (0, 1), grid = False,
        plot_kwargs = {'color': colors[i], 'linewidth': 0.6, 'linestyle': "--", 'dashes': (9, 3)})
    plot(s, np.abs(rho_exact(s, alpha, beta, L)), ax = ax, xlabel = r"$s$", ylabel = r"$|\rho(s)|$", logscale = False, 
        plot_kwargs = {'color': colors[i], 'linewidth': 0.6, 'linestyle': "-", 'label': rf'$\theta = {round(alpha - beta, 2)}$'}, xlim = (0, 20), ylim = (0, 1), grid = False)
    plt.legend()
legend = ax.legend(frameon=True, fancybox=False, facecolor="white", edgecolor="black", prop={"size": 13})
legend.get_frame().set_linewidth(0.7)
plt.rcParams.update({"font.size": 13})

# Save the figures
figure.savefig("figures/toy.svg", dpi = 300, bbox_inches = 'tight')

plt.show()