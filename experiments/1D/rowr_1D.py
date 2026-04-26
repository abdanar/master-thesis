import numpy as np
from fem.mesh import Mesh
from fem.femspace import FEMSpace
from fom.heat_fom import HeatProblem
from fom.oswr_heat import OSWRProblem
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

# Create a mesh - space mesh generation
vert1D = np.linspace(0, 1, 100)
mesh1D = Mesh(vertices = vert1D, dim = 1)

# Time domain definition (ntime = (T - t0)/dt + 1 for uniform time steps)
t0, T, ntime = 0.0, 1.0, 101
time_grid = np.linspace(t0, T, ntime)

# Linear Lagrange finite element space
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

# Solve the 1D Heat problem using nodal lifting and theta method with theta = 1 (implicit Euler)
heat_solution1D = problem1D.solve(time_grid = time_grid, lift = 'nodal', theta = 1)

# Define Reduced Schwarz problem with 2 subdomains and overlap of 1 layer of elements with version 1 of the decomposition algorithm
roswrproblem1D = ROSWRProblem(heat_problem = problem1D, n = 2, overlap = 1, version = 1)

# Compute POD modes for the full-order problem to be used in the Reduced Schwarz method
tparams = np.linspace(t0, T, 51) # time grid for snapshot collection (can be different from the time grid used for the full order solve)
pod1D = POD(heat_problem = problem1D, time_grid = tparams, lift = 'nodal', theta = 1, r = 10)

# Construct projection matrices for each subdomain by restricting the global snapshot matrix to the subdomain and performing SVD to extract the first r modes as projection matrix for the subdomain
projs1D = proj_matrices(femspace = femspace1D, roswrproblem = roswrproblem1D, pod = pod1D)

# Define a history configuration
subdomains = [1, 2]
time_indices = np.array([40, 80])
metrics = [MetricSpec(MetricType.ABSOLUTE_ERROR, SpatialMode.GLOBAL, TemporalMode.STATIC),
           MetricSpec(MetricType.CONVERGENCE_RATE, SpatialMode.SUBDOMAINS, TemporalMode.TIME)]
config = HistoryConfig(metrics = metrics, norm = NormType.L2, exact = exact1D, uh = heat_solution1D, time_indices = time_indices, subdomains = subdomains, mode = 'exact')

# Solve the problem using the Reduced Schwarz method with RAS, nodal lifting and theta method with theta = 1 (implicit Euler)
history1D, roswr_solution1D = roswrproblem1D.solve(projs = projs1D, time_grid = time_grid, theta = 1, lift = 'nodal', method = 'RAS', maxiter = 150, tol = 1e-9, histconfig = config)

# Error analysis (compute L2 error between reduced schwarz solution and exact solution, as well as between reduced schwarz solution and fem solution for each time step and report the maximum error across all time steps)
exactvals1D = exact1D(femspace1D.mesh.vertices[:, None], time_grid[None, :])
error1D = np.linalg.norm(roswr_solution1D - exactvals1D, axis = 0)
error_fem1D = np.linalg.norm(heat_solution1D - roswr_solution1D, axis = 0)
print("max error (reduced schwarz vs exact):", error1D.max())
print("max error (reduced schwarz vs fem):", error_fem1D.max())

# Visualize the error between the exact and ROSWR solutions at the mesh vertices
error1D_roswr = np.abs(exact1D(femspace1D.mesh.vertices[:, None], time_grid[None, :]) - roswr_solution1D) + 1e-14 # add small value to avoid log(0) issues
visualizer1D = visualize.SolutionVisualizer(mesh = femspace1D.mesh, u = error1D_roswr, dt = time_grid[1] - time_grid[0], femspace = femspace1D)
visualizer1D.plot(figsize = (6, 3), dpi = 150, logscale = True, ylabel = 'e', ymin = 1e-14, ymax = 1e-0, linewidth = 0.7, linestyle = '-', color = 'red', title = "Error between Exact and ROSWR Solutions")

# Visualize the error between the OSWR and ROSWR solutions at the mesh vertices
oswrproblem1D = OSWRProblem(heat_problem = problem1D, n = 2, overlap = 1, version = 1)
oswr_solution1D = oswrproblem1D.solve(time_grid = time_grid, theta = 1, lift = 'nodal', method = 'RAS', maxiter = 150, tol = 1e-9)
error1D_oswr = np.abs(exact1D(femspace1D.mesh.vertices[:, None], time_grid[None, :]) - oswr_solution1D) + 1e-14
styles_error = {'exactvsOSWR': {'label': 'OSWR', 'color': 'black', 'linestyle': '--', 'linewidth': 0.8}}
visualizer1D.plot(data = {'exactvsOSWR': error1D_oswr}, styles = styles_error, figsize = (6, 3), dpi = 150, logscale = True, 
                    ymin = 1e-14, ymax = 1e-0, linewidth = 0.8, slabel = "ROSWR", ylabel = 'e',
                    linestyle = '-', color = 'red', title = "OSWR vs ROSWR Error Comparison")

# Visualize the stored error between the exact and ROSWR solutions
global_history = history1D.values[MetricType.ABSOLUTE_ERROR]["global"] # shape (niter,)
subdomain_history = history1D.values[MetricType.CONVERGENCE_RATE]["subdomains"] # shape dictionary of shape {domainID: shape (niter, ntime)}
styles = {1: {'label': 'Subdomain 1', 'color': 'orange', 'linestyle': '-', 'linewidth': 0.6},
          2: {'label': 'Subdomain 2', 'color': 'blue', 'linestyle': '-', 'linewidth': 0.6}}
visualizer1D.plot_iteration(data = global_history, dpi = 300, ylabel = rf"$\| u_{{exact}} - u_{{ROSWR}} \|_{{L^2(0,{T};L^2({vert1D[0]}, {vert1D[-1]}))}}$", save_path="figures/1D/roswr/fig1D_global_rom(exact).png",
                            color = 'black', linestyle = '-', linewidth = 0.7)
for i in range(len(time_indices)):
    visualizer1D.plot_iteration(data = {domainID: subdomain_history[domainID][:, i] for domainID in subdomain_history}, dpi = 300, title = r"Convergence Rate", xlabel = r"Iteration ($k$)", 
                                ylabel = rf"$\dfrac{{\| u_{{exact}}(t_{{{time_indices[i]}}}) - u^{{k}}_{{ROSWR}}(t_{{{time_indices[i]}}}) \|_{{L^2}}}}{{\| u_{{exact}}(t_{{{time_indices[i]}}}) - u^{{k-1}}_{{ROSWR}}(t_{{{time_indices[i]}}}) \|_{{L^2}}}}$",
                                save_path=f"figures/1D/roswr/fig1D_subdomains_time{time_indices[i]}_rom(exact).png", styles = styles)