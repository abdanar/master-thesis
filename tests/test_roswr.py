import numpy as np
from fem.mesh import Mesh
from fem.femspace import FEMSpace
from fom.heat import HeatProblem
from rom.roswrelaxation import ROSWRProblem
import visualization.visualize as visualize 
from rom.pod import POD

# ---------------------------------
# 1D Example (Reduced Heat problem)
# ---------------------------------

# Create a mesh - space mesh generation
vert1D = np.linspace(0, 1, 100)
mesh1D = Mesh(vertices = vert1D, dim = 1)

# Time domain definition (ntime = (T - t0)/dt + 1 for uniform time steps)
t0 = 0.0
T = 1.0
ntime = 301
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

# Define Reduced Schwarz problem with 2 subdomains and overlap of 1 layer of elements with version 2 of the decomposition algorithm
roswrproblem1D = ROSWRProblem(femspace = femspace1D, t0 = t0, T = T, f = func1D, g = exact1D, h = h1D, n = 2, overlap = 1, version = 2)

# Compute POD modes for the full-order problem to be used in the Reduced Schwarz method
pod1D = POD(heat_problem = problem1D, ntime = 51, lift = 'nodal', theta = 0.5, r = 10)

# Construct projection matrices for each subdomain by restricting the global snapshot matrix to the subdomain and performing SVD to extract the first r modes as projection matrix for the subdomain
projs1D = {}
snapshot_matrix1D = np.zeros((femspace1D.nnodes, 51)) # shape (n_interior, n_time)
snapshot_matrix1D[femspace1D.interior_nodes,:] = pod1D.compute_snapshots()
for subdomain in roswrproblem1D.subdomains:
    subsnap = roswrproblem1D.restrict(snapshot_matrix1D, subdomain.domainID)[subdomain.interior_nodes(),:] # Restrict the snapshot matrix to the subdomain
    V, _, _ = np.linalg.svd(subsnap, full_matrices=False)
    projs1D[subdomain.domainID] = V[:, :pod1D.r] # Take the first r modes and use them as projection matrix for the subdomain

# Solve the problem using the Reduced Schwarz method with RAS, nodal lifting and theta method with theta = 0.5 (Crank-Nicolson)
roswr_solution1D = roswrproblem1D.solve(projs = projs1D, time_grid = time_grid, theta = 0.5, lift = 'nodal', method = 'RAS', omega = 1.0, maxiter = 150, tol = 1e-12)

# Solve the problem using the Reduced Schwarz method with RAS, nodal lifting and theta method with theta = 0.5 (Crank-Nicolson), while tracking convergence history
# roswr_solution1D = roswrproblem1D.solve(projs = projs1D, time_grid = time_grid, theta = 0.5, lift = 'nodal', method = 'RAS', omega = 1.0, maxiter = 100, tol = 1e-2, history = True, uh = heat_solution1D, exact = exact1D)

# Error analysis (compute L2 error between reduced schwarz solution and exact solution, as well as between reduced schwarz solution and fem solution for each time step and report the maximum error across all time steps)
error1D = np.linalg.norm(roswr_solution1D - exact1D(mesh1D.vertices[:, None], time_grid[None, :]), axis = 0)
error_fem1D = np.linalg.norm(heat_solution1D - roswr_solution1D, axis = 0)
print("max error (reduced schwarz vs exact):", error1D.max())
print("max error (reduced schwarz vs fem):", error_fem1D.max())

# Visualization
visualizer1D = visualize.SolutionVisualizer(mesh1D, roswr_solution1D)
visualizer1D.plot_convergence(error_history = roswrproblem1D.error_history, linewidth = 0.8, markersize = 3)

# ---------------------------------
# 2D Example (Reduced Heat problem)
# ---------------------------------

# Create a simple square mesh - space mesh generation
vertices = np.array([[0,0],[1,0],[1,1],[0,1]])
segments = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
segment_markers = np.array([1, 2, 3, 4])
mesh2D = Mesh(vertices = vertices, segments = segments, segment_markers = segment_markers, options = 'pqa0.01')

# Time domain definition (ntime = (T - t0)/dt + 1 for uniform time steps)
t0 = 0.0
T = 1.0
ntime = 301
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

# Define Reduced Schwarz problem with 3 subdomains and overlap of 1 layer of elements with version 2 of the decomposition algorithm
roswrproblem2D = ROSWRProblem(femspace = femspace2D, t0 = t0, T = T, f = func2D, g = exact2D, h = h2D, n = 3, overlap = 1, version = 2)

# Compute POD modes for the full-order problem to be used in the Reduced Schwarz method
pod2D = POD(heat_problem = problem2D, ntime = 51, lift = 'nodal', theta = 0.5, r = 10)

# Construct projection matrices for each subdomain by restricting the global snapshot matrix to the subdomain and performing SVD to extract the first r modes as projection matrix for the subdomain
projs2D = {}
snapshot_matrix2D = np.zeros((femspace2D.nnodes, 51)) # shape (n_interior, n_time)
snapshot_matrix2D[femspace2D.interior_nodes,:] = pod2D.compute_snapshots()
for subdomain in roswrproblem2D.subdomains:
    subsnap = roswrproblem2D.restrict(snapshot_matrix2D, subdomain.domainID)[subdomain.interior_nodes(),:] # Restrict the snapshot matrix to the subdomain
    V, _, _ = np.linalg.svd(subsnap, full_matrices=False)
    projs2D[subdomain.domainID] = V[:, :pod2D.r] # Take the first r modes and use them as projection matrix for the subdomain

# Solve the problem using the Reduced Schwarz method with RAS, nodal lifting and theta method with theta = 0.5 (Crank-Nicolson)
roswr_solution = roswrproblem2D.solve(projs = projs2D, time_grid = time_grid, theta = 0.5, lift = 'nodal', method = 'RAS', omega = 1.0, maxiter = 150, tol = 1e-12)

# Solve the problem using the Reduced Schwarz method with RAS, nodal lifting and theta method with theta = 0.5 (Crank-Nicolson), while tracking convergence history
# roswr_solution = roswrproblem2D.solve(projs = projs2D, time_grid = time_grid, theta = 0.5, lift = 'nodal', method = 'RAS', omega = 1.0, maxiter = 150, tol = 1e-9, history = True, uh = heat_solution2D, exact = exact2D)

# Error analysis (compute L2 error between reduced schwarz solution and exact solution, as well as between reduced schwarz solution and fem solution for each time step and report the maximum error across all time steps)
error2D = np.linalg.norm(roswr_solution - exact2D(femspace2D.mesh.vertices[:,0][:, None], femspace2D.mesh.vertices[:,1][:, None], time_grid[None, :]), axis = 0)
error_fem2D = np.linalg.norm(heat_solution2D - roswr_solution, axis = 0)
print("max error (reduced schwarz vs exact):", error2D.max())
print("max error (reduced schwarz vs fem):", error_fem2D.max())

# Visualization
styles = {1: {'color': 'black', 'linestyle': '-', 'linewidth': 0.8, 'marker': 'o', 'fillstyle': 'none', 'markersize': 6, 'markeredgewidth': 0.8},
          2: {'color': 'black', 'linestyle': '-', 'linewidth': 0.8, 'marker': 's', 'fillstyle': 'none', 'markersize': 6, 'markeredgewidth': 0.8},
          3: {'color': 'black', 'linestyle': '-', 'linewidth': 0.8, 'marker': '^', 'fillstyle': 'none', 'markersize': 6, 'markeredgewidth': 0.8}}
visualizer2D = visualize.SolutionVisualizer(femspace2D.mesh, roswr_solution)
visualizer2D.plot_convergence(error_history = roswrproblem2D.error_history, color = 'black', linestyle = '-', linewidth = 0.8, marker = 's', fillstyle = 'none', markersize = 6, markeredgewidth = 0.8)
visualizer2D.plot_convergence(error_history = roswrproblem2D.error_subdomains, styles = styles)