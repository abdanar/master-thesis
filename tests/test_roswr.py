import numpy as np
from fem.mesh import Mesh
from fem.femspace import FEMSpace
from fom.heat import HeatProblem
from rom.roswrelaxation import ROSWRProblem
import visualization.visualize as visualize 
from rom.pod import POD
from scipy.linalg import cholesky, solve_triangular

# ---------------------------------
# Helper function to construct projection matrices for each subdomain by restricting the global snapshot matrix to the
# ---------------------------------
def proj_matrices(femspace: FEMSpace, roswrproblem: ROSWRProblem, pod: POD, weighted: bool = False) -> dict[int, np.ndarray]:
    projs = {}
    snapshot_matrix = np.zeros((femspace.nnodes, pod.ntime)) # shape (n_interior, n_time)
    snapshot_matrix[femspace.interior_nodes,:] = pod.compute_snapshots()
    for subdomain in roswrproblem.subdomains:
        subsnap = roswrproblem.restrict(snapshot_matrix, subdomain.domainID)[subdomain.interior_nodes(),:] # Restrict the snapshot matrix to the subdomain
        if not weighted:
            V, _, _ = np.linalg.svd(subsnap, full_matrices=False)
            projs[subdomain.domainID] = V[:, :pod.r]
        else:
            L = cholesky(roswrproblem.subproblems[subdomain.domainID].stiffness_matrix_II, lower=True)
            V, S, _ = np.linalg.svd(L @ subsnap, full_matrices=False)
            projs[subdomain.domainID] = solve_triangular(L, V[:, :pod.r], lower=True)
    return projs

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

# Define Reduced Schwarz problem with 3 subdomains and overlap of 1 layer of elements with version 1 of the decomposition algorithm
roswrproblem1D = ROSWRProblem(femspace = femspace1D, t0 = t0, T = T, f = func1D, g = exact1D, h = h1D, n = 3, overlap = 1, version = 1)

# Compute POD modes for the full-order problem to be used in the Reduced Schwarz method
pod1D = POD(heat_problem = problem1D, ntime = 51, lift = 'nodal', theta = 0.5, r = 10)

# Construct projection matrices for each subdomain by restricting the global snapshot matrix to the subdomain and performing SVD to extract the first r modes as projection matrix for the subdomain
projs1D = proj_matrices(femspace = femspace1D, roswrproblem = roswrproblem1D, pod = pod1D, weighted = False)

# Solve the problem using the Reduced Schwarz method with RAS, nodal lifting and theta method with theta = 0.5 (Crank-Nicolson)
roswr_solution1D = roswrproblem1D.solve(projs = projs1D, time_grid = time_grid, theta = 0.5, lift = 'nodal', method = 'RAS', omega = 1.0, maxiter = 150, tol = 1e-12)

# Solve the problem using the Reduced Schwarz method with RAS, nodal lifting and theta method with theta = 0.5 (Crank-Nicolson), while tracking convergence history
# roswr_solution1D = roswrproblem1D.solve(projs = projs1D, time_grid = time_grid, theta = 0.5, lift = 'nodal', method = 'RAS', omega = 1.0, maxiter = 100, tol = 1e-2, history = True, uh = heat_solution1D, exact = exact1D)

# Error analysis (compute L2 error between reduced schwarz solution and exact solution, as well as between reduced schwarz solution and fem solution for each time step and report the maximum error across all time steps)
exactvals1D = exact1D(mesh1D.vertices[:, None], time_grid[None, :])
error1D = np.linalg.norm(roswr_solution1D - exactvals1D, axis = 0)
error_fem1D = np.linalg.norm(heat_solution1D - roswr_solution1D, axis = 0)
print("max error (reduced schwarz vs exact):", error1D.max())
print("max error (reduced schwarz vs fem):", error_fem1D.max())

# Visualization
solvisualizer1D = visualize.SolutionVisualizer(mesh1D, roswr_solution1D)
solvisualizer1D.plot_convergence(error_history = roswrproblem1D.error_history, linewidth = 0.8, markersize = 3)
errvisualizer1D = visualize.SolutionVisualizer(mesh1D, np.abs(roswr_solution1D - exactvals1D)[:, 25])
errvisualizer1D.visualize(color = 'red', linestyle = '-', linewidth = 0.8, logscale=True, xlabel="x", ylabel="Error", title="Error at t = {}".format(time_grid[25]))

# ---------------------------------
# 2D Example (Reduced Heat problem)
# ---------------------------------

# Create a simple square mesh - space mesh generation
vertices = np.array([[0,0],[1,0],[1,1],[0,1]])
segments = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
segment_markers = np.array([1, 2, 3, 4])
mesh2D = Mesh(vertices = vertices, segments = segments, segment_markers = segment_markers, options = 'pqa0.001')

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
projs2D = proj_matrices(femspace = femspace2D, roswrproblem = roswrproblem2D, pod = pod2D, weighted = False)

# Solve the problem using the Reduced Schwarz method with RAS, nodal lifting and theta method with theta = 0.5 (Crank-Nicolson)
roswr_solution2D = roswrproblem2D.solve(projs = projs2D, time_grid = time_grid, theta = 0.5, lift = 'nodal', method = 'RAS', omega = 1.0, maxiter = 150, tol = 1e-12)

# Solve the problem using the Reduced Schwarz method with RAS, nodal lifting and theta method with theta = 0.5 (Crank-Nicolson), while tracking convergence history
# roswr_solution = roswrproblem2D.solve(projs = projs2D, time_grid = time_grid, theta = 0.5, lift = 'nodal', method = 'RAS', omega = 1.0, maxiter = 150, tol = 1e-9, history = True, uh = heat_solution2D, exact = exact2D)

# Error analysis (compute L2 error between reduced schwarz solution and exact solution, as well as between reduced schwarz solution and fem solution for each time step and report the maximum error across all time steps)
exactvals2D = exact2D(femspace2D.mesh.vertices[:,0][:, None], femspace2D.mesh.vertices[:,1][:, None], time_grid[None, :])
error2D = np.linalg.norm(roswr_solution2D - exactvals2D, axis = 0)
error_fem2D = np.linalg.norm(heat_solution2D - roswr_solution2D, axis = 0)
print("max error (reduced schwarz vs exact):", error2D.max())
print("max error (reduced schwarz vs fem):", error_fem2D.max())

# Visualization
styles = {1: {'color': 'black', 'linestyle': '-', 'linewidth': 0.8, 'marker': 'o', 'fillstyle': 'none', 'markersize': 6, 'markeredgewidth': 0.8},
          2: {'color': 'black', 'linestyle': '-', 'linewidth': 0.8, 'marker': 's', 'fillstyle': 'none', 'markersize': 6, 'markeredgewidth': 0.8},
          3: {'color': 'black', 'linestyle': '-', 'linewidth': 0.8, 'marker': '^', 'fillstyle': 'none', 'markersize': 6, 'markeredgewidth': 0.8}}
visualizer2D = visualize.SolutionVisualizer(femspace2D.mesh, roswr_solution2D)
visualizer2D.plot_convergence(error_history = roswrproblem2D.error_history, color = 'black', linestyle = '-', linewidth = 0.8, marker = 's', fillstyle = 'none', markersize = 6, markeredgewidth = 0.8)
#visualizer2D.plot_convergence(error_history = roswrproblem2D.error_subdomains, styles = styles)
errvisualizer2D = visualize.SolutionVisualizer(femspace2D.mesh, np.abs(roswr_solution2D - exactvals2D)[:, 40])
errvisualizer2D.visualize(color = 'red', linestyle = '-', linewidth = 0.8, logscale=True, xlabel="x", ylabel="Error", title="Error at t = {}".format(time_grid[40]))
errvisualizer2D.visualize_3d()


# def exact1D(x, t):
#     return np.exp(-t)*np.sin(np.pi*x) + x

# def func1D(x, t):
#     return np.exp(-t)*(np.pi**2 - 1)*np.sin(np.pi*x)

# def h1D(x):
#     return np.sin(np.pi*x) + x

# def exact2D(x, y, t):
#     return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y) + x

# def func2D(x, y, t):
#     return np.exp(-t) * (2*np.pi**2 - 1) * np.sin(np.pi * x) * np.sin(np.pi * y)

# def h2D(x, y):
#     return np.sin(np.pi * x) * np.sin(np.pi * y) + x