import numpy as np
from fem.mesh import Mesh
from fem.femspace import FEMSpace
from fom.heat_fom import HeatProblem
from rom.roswr_heat import ROSWRProblem
import visualization.visualize as visualize 
from rom.pod import POD
from utils.history import HistoryConfig, MetricSpec, MetricType, SpatialMode, TemporalMode
from utils.errornorms import NormType
from utils.logger import configure_logging
configure_logging(level="info")

# Helper function to construct projection matrices for each subdomain by restricting the global snapshot matrix to the subdomain
def proj_matrices(femspace: FEMSpace, roswrproblem: ROSWRProblem, pod: POD) -> dict[int, np.ndarray]:
    projs = {}
    snapshot_matrix = np.zeros((femspace.nnodes, pod.ntime)) # shape (n_interior, n_time)
    snapshot_matrix[femspace.interior_nodes,:] = pod.compute_snapshots()
    for subdomain_id, subdomain in roswrproblem.subdomains.items():
        subsnap = roswrproblem.restrict(snapshot_matrix, subdomain_id)[subdomain.interior_nodes(),:] # Restrict the snapshot matrix to the subdomain
        V, _, _ = np.linalg.svd(subsnap, full_matrices=False)
        projs[subdomain_id] = V[:, :pod.r]
    return projs

# Create a simple square mesh - space mesh generation
vertices = np.array([[0,0],[1,0],[1,1],[0,1]])
mesh2D = Mesh(vertices = vertices, options = 'qa0.01')

# Time domain definition (ntime = (T - t0)/dt + 1 for uniform time steps)
t0, T, ntime = 0.0, 1.0, 101
time_grid = np.linspace(t0, T, ntime)

# Linear Lagrange finite element space
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

# Define Reduced Schwarz problem with 2 subdomains and overlap of 1 layer of elements with version 1 of the decomposition algorithm
roswrproblem2D = ROSWRProblem(femspace = femspace2D, t0 = t0, T = T, f = func2D, g = exact2D, h = h2D, n = 2, overlap = 1, version = 1)

# Compute POD modes for the full-order problem to be used in the Reduced Schwarz method
pod2D = POD(heat_problem = problem2D, ntime = 51, lift = 'nodal', theta = 0.5, r = 10)

# Construct projection matrices for each subdomain by restricting the global snapshot matrix to the subdomain and performing SVD to extract the first r modes as projection matrix for the subdomain
projs2D = proj_matrices(femspace = femspace2D, roswrproblem = roswrproblem2D, pod = pod2D)

# Define a history configuration
subdomains = [1, 2]
time_indices = np.array([30, 60])
metrics = [MetricSpec(MetricType.ABSOLUTE_ERROR, SpatialMode.GLOBAL, TemporalMode.STATIC),
           MetricSpec(MetricType.CONVERGENCE_RATE, SpatialMode.SUBDOMAINS, TemporalMode.TIME)]
config = HistoryConfig(metrics = metrics, norm = NormType.L2, exact = exact2D, uh = heat_solution2D, time_indices = time_indices, subdomains = subdomains, mode = 'exact')

# Solve the problem using the Reduced Schwarz method with RAS, nodal lifting and theta method with theta = 0.5 (Crank-Nicolson)
history2D, roswr_solution2D = roswrproblem2D.solve(projs = projs2D, time_grid = time_grid, theta = 0.5, lift = 'nodal', method = 'RAS', maxiter = 100, tol = 1e-9, histconfig = config)

# Error analysis (compute L2 error between reduced schwarz solution and exact solution, as well as between reduced schwarz solution and fem solution for each time step and report the maximum error across all time steps)
exactvals2D = exact2D(femspace2D.mesh.vertices[:,0][:, None], femspace2D.mesh.vertices[:,1][:, None], time_grid[None, :])
error2D = np.linalg.norm(roswr_solution2D - exactvals2D, axis = 0)
error_fem2D = np.linalg.norm(heat_solution2D - roswr_solution2D, axis = 0)
print("max error (reduced schwarz vs exact):", error2D.max())
print("max error (reduced schwarz vs fem):", error_fem2D.max())

# Mesh visualization
visualizer2D = visualize.MeshVisualizer(mesh2D)
visualizer2D.plot_subdomains(subdomains = roswrproblem2D.subdomains, membership = roswrproblem2D.membership, 
                            show_vertex_markers = False, show_node_numbers = False, show_element_numbers = False)

# Solution visualization
global_history = history2D.values[MetricType.ABSOLUTE_ERROR]["global"] # shape (niter,)
subdomain_history = history2D.values[MetricType.CONVERGENCE_RATE]["subdomains"] # shape dictionary of shape {domainID: shape (niter, ntime)}
visualizer2D = visualize.SolutionVisualizer(femspace2D.mesh, roswr_solution2D)
styles = {1: {'color': 'orange', 'linestyle': '-', 'linewidth': 0.8},
          2: {'color': 'blue', 'linestyle': '-', 'linewidth': 0.8}}
visualizer2D.plot_convergence(error_history = global_history, ylabel = r"$\| u_{exact} - u_{ROSWR} \|_{L^2}$", save_path="figures/2D/fig2D_global_rom(exact).png",
                              color = 'black', linestyle = '-', linewidth = 0.8)
for i in range(len(time_indices)):
    visualizer2D.plot_convergence(error_history = {domainID: subdomain_history[domainID][:, i] for domainID in subdomain_history}, title = r"Convergence Rate", xlabel = r"Iteration ($k$)", 
                                  ylabel = rf"$\dfrac{{\| u_{{exact}}(t_{{{time_indices[i]}}}) - u^{{k}}_{{ROSWR}}(t_{{{time_indices[i]}}}) \|_{{L^2}}}}{{\| u_{{exact}}(t_{{{time_indices[i]}}}) - u^{{k-1}}_{{ROSWR}}(t_{{{time_indices[i]}}}) \|_{{L^2}}}}$",
                                  save_path=f"figures/2D/fig2D_subdomains_time{time_indices[i]}_rom(exact).png", styles = styles)


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