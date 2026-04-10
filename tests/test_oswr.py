import numpy as np
from fem.mesh import Mesh
from fem.femspace import FEMSpace
from fom.heat import HeatProblem
from fom.oswrelaxation import OSWRProblem
import visualization.visualize as visualize 

# ----------------------------
# 1D Example (Heat problem)
# ----------------------------

# Create a mesh - space mesh generation
vert1D = np.linspace(0, 1, 100)
mesh1D = Mesh(vertices = vert1D, dim = 1)

# Time domain definition (ntime = (T - t0)/dt + 1 for uniform time steps)
t0 = 0.0
T = 1.0
ntime = 101
time_grid = np.linspace(t0, T, ntime)

# Finite element space of degree 1
femspace1D = FEMSpace(mesh1D, domain = 'interval', degree = 1)

# Define the exact solution
def exact1D(x, t):
    return np.exp(-t)*(x**2 + x)

# Define the source function
def func1D(x, t):
    return -np.exp(-t)*(x**2 + x) - 2*np.exp(-t)

# Define the initial condition function
def h1D(x):
    return np.exp(-t0)*(x**2 + x)

# Define 1D Heat problem
problem1D = HeatProblem(femspace = femspace1D, t0 = t0, T = T, f = func1D, g = exact1D, h = h1D)

# Solve the 1D Heat problem using nodal lifting and theta method with theta = 0.5 (Crank-Nicolson)
heat_solution1D = problem1D.solve(time_grid = time_grid, lift = 'nodal', theta = 0.5)

# Define Schwarz problem with 2 subdomains and overlap of 1 layer of elements with version 2 of the decomposition algorithm
oswrproblem1D = OSWRProblem(femspace = femspace1D, t0 = t0, T = T, f = func1D, g = exact1D, h = h1D, n = 2, overlap = 1, version = 2)

# Solve the problem using the Schwarz method with RAS, nodal lifting and theta method with theta = 0.5 (Crank-Nicolson)
oswr_solution1D = oswrproblem1D.solve(time_grid = time_grid, theta = 0.5, lift = 'nodal', method = 'RAS', omega = 1.0, maxiter = 150, tol = 1e-12)

# Solve the problem using the Schwarz method with RAS, nodal lifting and theta method with theta = 0.5 (Crank-Nicolson), while tracking convergence history
# oswr_solution1D = oswrproblem1D.solve(time_grid = time_grid, theta = 0.5, lift = 'nodal', method = 'RAS', omega = 1.0, maxiter = 100, tol = 1e-3, history = True, uh = poisson_solution1D, exact = exact1D)

# Error analysis (compute L2 error between schwarz solution and exact solution, as well as between schwarz solution and fem solution for each time step and report the maximum error across all time steps)
error1D = np.linalg.norm(oswr_solution1D - exact1D(mesh1D.vertices[:, None], time_grid[None, :]), axis = 0)
error_fem1D = np.linalg.norm(heat_solution1D - oswr_solution1D, axis = 0)
print("max error (schwarz vs exact):", error1D.max())
print("max error (schwarz vs fem):", error_fem1D.max())

# Visualization
visualizer1D = visualize.SolutionVisualizer(mesh1D, oswr_solution1D)
visualizer1D.plot_convergence(error_history = oswrproblem1D.error_history, linewidth = 0.8, markersize = 3)

# ----------------------------
# 2D Example (Heat problem)
# ----------------------------

# Create a simple square mesh - space mesh generation
vertices = np.array([[0,0],[1,0],[1,1],[0,1]])
segments = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
segment_markers = np.array([1, 2, 3, 4])
mesh2D = Mesh(vertices = vertices, segments = segments, segment_markers = segment_markers, options = 'pqa0.01')

# Time domain definition (ntime = (T - t0)/dt + 1 for uniform time steps)
t0 = 0.0
T = 3.0
ntime = 101
time_grid = np.linspace(t0, T, ntime)

# Finite element space of degree 1
femspace2D = FEMSpace(mesh2D, degree = 1)

# Define the exact solution
def exact2D(x, y, t):
    return np.exp(-t)*(x**2 + y**2)

# Define a source function
def func2D(x, y, t):
    return -np.exp(-t)*(x**2 + y**2) - 4*np.exp(-t)

# Define the initial condition function
def h2D(x, y):
    return np.exp(-t0)*(x**2 + y**2)

# Define 2D Heat problem
problem2D = HeatProblem(femspace = femspace2D, t0 = t0, T = T, f = func2D, g = exact2D, h = h2D)

# Solve the 2D Heat problem using nodal lifting and theta method with theta = 0.5 (Crank-Nicolson)
heat_solution2D = problem2D.solve(time_grid = time_grid, lift = 'nodal', theta = 0.5)

# Define Schwarz problem with 3 subdomains and overlap of 1 layer of elements with version 2 of the decomposition algorithm
oswrproblem2D = OSWRProblem(femspace = femspace2D, t0 = t0, T = T, f = func2D, g = exact2D, h = h2D, n = 3, overlap = 1, version = 2)

# Solve the problem using the Schwarz method with RAS, nodal lifting and theta method with theta = 0.5 (Crank-Nicolson)
oswr_solution = oswrproblem2D.solve(time_grid = time_grid, theta = 0.5, lift = 'nodal', method = 'RAS', omega = 1.0, maxiter = 150, tol = 1e-12)

# Solve the problem using the Schwarz method with RAS, nodal lifting and theta method with theta = 0.5 (Crank-Nicolson), while tracking convergence history
# oswr_solution = oswrproblem2D.solve(time_grid = time_grid, theta = 0.5, lift = 'nodal', method = 'RAS', omega = 1.0, maxiter = 100, tol = 1e-2, history = True, uh = heat_solution2D, exact = exact2D)

# Error analysis (compute L2 error between schwarz solution and exact solution, as well as between schwarz solution and fem solution for each time step and report the maximum error across all time steps)
error2D = np.linalg.norm(oswr_solution - exact2D(femspace2D.mesh.vertices[:,0][:, None], femspace2D.mesh.vertices[:,1][:, None], time_grid[None, :]), axis = 0)
error_fem2D = np.linalg.norm(heat_solution2D - oswr_solution, axis = 0)
print("max error (schwarz vs exact):", error2D.max())
print("max error (schwarz vs fem):", error_fem2D.max())

# Visualization
styles = {1: {'color': 'black', 'linestyle': '-', 'linewidth': 0.8, 'marker': 'o', 'fillstyle': 'none', 'markersize': 6, 'markeredgewidth': 0.8},
          2: {'color': 'black', 'linestyle': '-', 'linewidth': 0.8, 'marker': 's', 'fillstyle': 'none', 'markersize': 6, 'markeredgewidth': 0.8},
          3: {'color': 'black', 'linestyle': '-', 'linewidth': 0.8, 'marker': '^', 'fillstyle': 'none', 'markersize': 6, 'markeredgewidth': 0.8}}
visualizer2D = visualize.SolutionVisualizer(femspace2D.mesh, oswr_solution)
visualizer2D.plot_convergence(error_history = oswrproblem2D.error_history, color = 'black', linestyle = '-', linewidth = 0.8, marker = 's', fillstyle = 'none', markersize = 6, markeredgewidth = 0.8)
visualizer2D.plot_convergence(error_history = oswrproblem2D.error_subdomains, styles = styles)