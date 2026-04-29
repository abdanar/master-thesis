import numpy as np
import matplotlib.pyplot as plt
from fem.femspace import FEMSpace
from fem.mesh import Mesh
from fom.heat_fom import HeatProblem
from rom.roswr_heat import ROSWRProblem
from fom.oswr_heat import OSWRProblem
from visualization.visualize import plot_wireframe, plot
from fem.linearsolver import DirectSolver
from utils.logger import configure_logging
configure_logging(level="info")

# Helper function to construct projection matrices for each subdomain by restricting the global snapshot matrix to the subdomain
def proj_matrices(r: int, ntime: int, femspace: FEMSpace, roswrproblem: ROSWRProblem, lift, theta: float, solver) -> dict[int, np.ndarray]:
    projs = {}
    snapshot_matrix = np.zeros((femspace.nnodes, ntime))
    time_grid = np.linspace(roswrproblem.heat_problem.t0, roswrproblem.heat_problem.T, ntime)
    snapshot_matrix[femspace.interior_nodes,:] = roswrproblem.heat_problem.solve(time_grid = time_grid, lift = lift, theta = theta, solver = solver, 
                                                g_new = boundary_condition, homogeneous = True)
    for subdomain_id, subdomain in roswrproblem.subdomains.items():
        subsnap = roswrproblem.restrict(snapshot_matrix, subdomain_id)[subdomain.interior_nodes(),:] # Restrict the snapshot matrix to the subdomain
        V, _, _ = np.linalg.svd(subsnap, full_matrices = True)
        projs[subdomain_id] = V[:, :r]
    return projs 

# Create a mesh - space mesh generation
vert1D = np.linspace(0, 1, 101)
mesh1D = Mesh(vertices = vert1D, dim = 1)

# Time domain definition (dt = 0.01 for uniform time steps)
t0, T, ntime = 0, 1, 101
time_grid = np.linspace(t0, T, ntime)

# Linear Lagrange finite element space
femspace1D = FEMSpace(mesh = mesh1D, domain = 'interval', degree = 1)

# Define the source function
def source_function(x, t):
    return 2*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*t) + 8*np.pi**2*np.cos(2*np.pi*x)*np.cos(2*np.pi*t)

# Define the boundary condition
def boundary_condition(x, t):
    return np.sin(np.pi*x)*np.sin(np.pi*t) + np.cos(2*np.pi*x)*np.cos(2*np.pi*t)

# Define the initial condition
def initial_condition(x):
    return np.cos(2*np.pi*x)

# Define 1D Heat problem
problem1D = HeatProblem(femspace = femspace1D, t0 = t0, T = T, f = source_function, g = boundary_condition, h = initial_condition)

# Solve the 1D Heat problem using nodal lifting and theta method with theta = 1 (Backward Euler)
heat_solution1D = problem1D.solve(time_grid = time_grid, lift = 'nodal', theta = 1)

# Plot the solution of the heat equation
global_fig, global_ax = plot_wireframe(vert1D, time_grid, heat_solution1D.T, xlabel=r'$x$', ylabel=r'$t$', title = r'Solution of the 1D Heat Equation', xlim=(0, 1), ylim=(t0, T), azim = 255, elev = 15)
global_ax.zaxis.set_rotate_label(False)
global_ax.set_zlabel(r'$u(x,t)$', rotation = 90)

# Define the Reduced Overlapping Schwarz Waveform Relaxation (ROSWR) problem with 2 subdomains and an overlap of 10 layers
roswr_problem1D = ROSWRProblem(heat_problem = problem1D, n = 2, overlap = 10)

# Construct projection matrices for each subdomain by restricting the global snapshot matrix to the subdomain and performing SVD to extract the first r modes as projection matrix for the subdomain
projs1D = proj_matrices(r = 10, ntime = 51, femspace = femspace1D, roswrproblem = roswr_problem1D, lift = 'nodal', theta = 1, solver = DirectSolver())

# Solve the ROSWR problem using nodal lifting and theta method with theta = 1 (Backward Euler), and store the subdomain solutions for iterations
iterations = list(range(1, 21))
roswr_solution1D = roswr_problem1D.solve(projs = projs1D, time_grid = time_grid, lift = 'nodal', theta = 1, maxiter = 25, tol = 1e-14, store_solution = iterations)

# Extract the subdomain solutions for the iterations n = 1, 2, 3, 4
iterates = roswr_problem1D.iterates

# Plot the subdomain solutions and errors for iterations n = 1, 2, 3, 4
subdomain_fig, axes = plt.subplots(4, 2, figsize=(7, 14), subplot_kw={"projection": "3d"})
for i, iter in enumerate([1, 2, 3, 4]):
    # Extract the subdomain solutions for the current iteration
    Z1 = np.full_like(heat_solution1D, np.nan, dtype=float)
    Z2 = np.full_like(heat_solution1D, np.nan, dtype=float)
    Z1[roswr_problem1D.ltog[1], :] = iterates[iter][1]
    Z2[roswr_problem1D.ltog[2], :] = iterates[iter][2]

    # Compute the error between the ROSWR solution and the heat solution for the current iteration
    err1 = heat_solution1D[roswr_problem1D.ltog[1], :] - iterates[iter][1]
    err2 = heat_solution1D[roswr_problem1D.ltog[2], :] - iterates[iter][2]
    Z3 = np.full_like(heat_solution1D, np.nan, dtype=float)
    Z4 = np.full_like(heat_solution1D, np.nan, dtype=float)
    Z3[roswr_problem1D.ltog[1], :] = err1
    Z4[roswr_problem1D.ltog[2], :] = err2

    # Plot the subdomain solutions and errors for the current iteration
    plot_wireframe(vert1D, time_grid, Z1.T, ax = axes[i, 0], xlabel=r'$x$', ylabel=r'$t$', xlim=(0, 1), ylim=(t0, T), azim = 255, elev = 15)
    plot_wireframe(vert1D, time_grid, Z2.T, ax = axes[i, 0], xlabel=r'$x$', ylabel=r'$t$', xlim=(0, 1), ylim=(t0, T), azim = 255, elev = 15)
    plot_wireframe(vert1D, time_grid, Z3.T, ax = axes[i, 1], xlabel=r'$x$', ylabel=r'$t$', xlim=(0, 1), ylim=(t0, T), azim = 255, elev = 15)
    plot_wireframe(vert1D, time_grid, Z4.T, ax = axes[i, 1], xlabel=r'$x$', ylabel=r'$t$', xlim=(0, 1), ylim=(t0, T), azim = 255, elev = 15)

# Compute absolute error between the ROSWR solution and the heat solution vs iteration number for different r values
err_data = {}
miniter = 0
r_iterations = list(range(1, 101))
r_values = [2, 5, 7, 10, 15]
for r in r_values:
    projs = proj_matrices(r = r, ntime = 51, femspace = femspace1D, roswrproblem = roswr_problem1D, lift = 'nodal', theta = 1, solver = DirectSolver())
    roswr_solution1D = roswr_problem1D.solve(projs = projs, time_grid = time_grid, lift = 'nodal', theta = 1, maxiter = 100, tol = 1e-12, store_solution = r_iterations)
    r_iterates = roswr_problem1D.iterates
    niter = len(r_iterates)
    miniter = min(miniter, niter) if miniter > 0 else niter
    err_data[r] = np.zeros(niter)
    for j in range(1, niter+1):
        err1 = np.abs(heat_solution1D[roswr_problem1D.ltog[1], :] - r_iterates[j][1])
        err2 = np.abs(heat_solution1D[roswr_problem1D.ltog[2], :] - r_iterates[j][2])
        err_data[r][j-1] = max(err1.max(), err2.max())

# Solve the OSWR problem using nodal lifting and theta method with theta = 1 (Backward Euler), and store the subdomain solutions for iterations
oswr_problem1D = OSWRProblem(heat_problem = problem1D, n = 2, overlap = 10)
oswr_solution1D = oswr_problem1D.solve(time_grid = time_grid, lift = 'nodal', theta = 1, maxiter = 80, tol = 1e-12, store_solution = r_iterations)
oswr_iterates = oswr_problem1D.iterates
oswr_iter = len(oswr_iterates)
oswr_data = np.zeros(oswr_iter)
for i in range(1, oswr_iter+1):
    err1 = np.abs(heat_solution1D[oswr_problem1D.ltog[1], :] - oswr_iterates[i][1])
    err2 = np.abs(heat_solution1D[oswr_problem1D.ltog[2], :] - oswr_iterates[i][2])
    oswr_data[i-1] = max(err1.max(), err2.max())

# Plot the error vs iteration number for different r values
#miniter = min(miniter, oswr_iter)
iterates_fig, ax = plt.subplots(figsize=(7, 5))
colors = ['blue', 'orange', 'green', 'black', 'purple']
for r in r_values:
    plot(r_iterations[:miniter], err_data[r][:miniter], ax = ax, plot_kwargs = {'linewidth': 0.7, 'linestyle':'-', 'color': colors[r_values.index(r)]})
plot(r_iterations[:oswr_iter], oswr_data[:oswr_iter], ax = ax, plot_kwargs = {'linewidth': 0.7, 'linestyle':'--', 'color': 'red'})
ax.set_xlabel('Iteration')
ax.set_ylabel('Error')
ax.legend([rf'$r = {r}$' for r in r_values] + ['OSWR'])

# Save the figures
plt.tight_layout()
subdomain_fig.savefig(f"figures/1D/roswr/ex3/subdomains_{T}.png", dpi = 300, bbox_inches = 'tight')
iterates_fig.savefig(f"figures/1D/roswr/ex3/iterates_{T}.png", dpi = 300, bbox_inches = 'tight')
plt.show()
