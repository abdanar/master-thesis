import numpy as np
import matplotlib.pyplot as plt
from fem.femspace import FEMSpace
from fem.mesh import Mesh
from fom.heat_fom import HeatProblem
from fom.oswr_heat import OSWRProblem
from visualization.visualize import plot_wireframe, plot

from utils.logger import configure_logging
configure_logging(level="info")

# Create a mesh - space mesh generation
vert1D = np.linspace(0, 1, 101)
mesh1D = Mesh(vertices = vert1D, dim = 1)

# Time domain definition (dt = 0.01 for uniform time steps)
t0, T, ntime = 0, 0.05, 6
time_grid = np.linspace(t0, T, ntime)

# Linear Lagrange finite element space
femspace1D = FEMSpace(mesh = mesh1D, domain = 'interval', degree = 1)

# Define the source function
def source_function(x, t):
    return 5*np.exp(-(t - 2)**2 - (x - 0.25)**2)

# Define the boundary condition
boundary_condition = np.vstack([np.zeros(ntime), np.exp(-time_grid)])

# Define the initial condition
def initial_condition(x):
    return x**2

# Define 1D Heat problem
problem1D = HeatProblem(femspace = femspace1D, t0 = t0, T = T, f = source_function, g = boundary_condition, h = initial_condition)

# Solve the 1D Heat problem using nodal lifting and theta method with theta = 1 (Backward Euler)
heat_solution1D = problem1D.solve(time_grid = time_grid, lift = 'nodal', theta = 1)

# Plot the solution of the heat equation
global_fig, global_ax = plot_wireframe(vert1D, time_grid, heat_solution1D.T, xlabel=r'$x$', ylabel=r'$t$', xlim=(0, 1), ylim=(t0, T))

# Define the Overlapping Schwarz Waveform Relaxation (OSWR) problem with 2 subdomains and an overlap of 10 layers
oswr_problem1D = OSWRProblem(heat_problem = problem1D, n = 2, overlap = 10)

# Solve the OSWR problem using nodal lifting and theta method with theta = 1 (Backward Euler), and store the subdomain solutions for iterations
iterations = list(range(1, 21))
oswr_solution1D = oswr_problem1D.solve(time_grid = time_grid, lift = 'nodal', theta = 1, maxiter = 25, tol = 1e-14, store_solution = iterations)

# Extract the subdomain solutions for the iterations n = 1, 2, 3, 4
iterates = oswr_problem1D.iterates

# Plot the subdomain solutions and errors for iterations n = 1, 2, 3, 4
subdomain_fig, axes = plt.subplots(4, 2, figsize=(7, 14), subplot_kw={"projection": "3d"})
for i, iter in enumerate([1, 2, 3, 4]):
    # Extract the subdomain solutions for the current iteration
    Z1 = np.full_like(heat_solution1D, np.nan, dtype=float)
    Z2 = np.full_like(heat_solution1D, np.nan, dtype=float)
    Z1[oswr_problem1D.ltog[1], :] = iterates[iter][1]
    Z2[oswr_problem1D.ltog[2], :] = iterates[iter][2]

    # Compute the error between the OSWR solution and the heat solution for the current iteration
    err1 = heat_solution1D[oswr_problem1D.ltog[1], :] - iterates[iter][1]
    err2 = heat_solution1D[oswr_problem1D.ltog[2], :] - iterates[iter][2]
    Z3 = np.full_like(heat_solution1D, np.nan, dtype=float)
    Z4 = np.full_like(heat_solution1D, np.nan, dtype=float)
    Z3[oswr_problem1D.ltog[1], :] = err1
    Z4[oswr_problem1D.ltog[2], :] = err2

    # Plot the subdomain solutions and errors for the current iteration
    plot_wireframe(vert1D, time_grid, Z1.T, ax = axes[i, 0], xlabel=r'$x$', ylabel=r'$t$', xlim=(0, 1), ylim=(t0, T))
    plot_wireframe(vert1D, time_grid, Z2.T, ax = axes[i, 0], xlabel=r'$x$', ylabel=r'$t$', xlim=(0, 1), ylim=(t0, T))
    plot_wireframe(vert1D, time_grid, Z3.T, ax = axes[i, 1], xlabel=r'$x$', ylabel=r'$t$', xlim=(0, 1), ylim=(t0, T))
    plot_wireframe(vert1D, time_grid, Z4.T, ax = axes[i, 1], xlabel=r'$x$', ylabel=r'$t$', xlim=(0, 1), ylim=(t0, T))

# Compute absolute error between the OSWR solution and the heat solution vs iteration number
iter_values = list(iterates.keys())
err_data = np.zeros(len(iter_values))
for j in iter_values:
    err1 = np.abs(heat_solution1D[oswr_problem1D.ltog[1], :] - iterates[j][1])
    err2 = np.abs(heat_solution1D[oswr_problem1D.ltog[2], :] - iterates[j][2])
    err = max(err1.max(), err2.max())
    err_data[j-1] = err

# Plot the error vs iteration number
iterates_fig, _ = plot(iter_values, err_data, xlabel = 'Iteration', ylabel = 'Error', 
    xticks = iter_values, plot_kwargs = {'linewidth': 0.7, 'linestyle':'-', 'color': 'blue'})

# Save the figures
plt.tight_layout()
global_fig.savefig(f"figures/1D/fem/ex2/fem_{T}.png", dpi = 300, bbox_inches = 'tight')
subdomain_fig.savefig(f"figures/1D/oswr/ex2/subdomains_{T}.png", dpi = 300, bbox_inches = 'tight')
iterates_fig.savefig(f"figures/1D/oswr/ex2/iterates_{T}.png", dpi = 300, bbox_inches = 'tight')
plt.show()
