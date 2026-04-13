import numpy as np
from utils.errornorms import ErrorNorms
from fem.mesh import Mesh
from fem.femspace import FEMSpace
from fom.heat import HeatProblem
from fom.oswrelaxation import OSWRProblem
from rom.roswrelaxation import ROSWRProblem
from rom.rheat import ReducedHeatProblem
import visualization.visualize as visualize 
from rom.pod import POD
from scipy.linalg import cholesky, solve_triangular

# ---------------------------------
# Helper function to construct projection matrices for each subdomain by restricting the global snapshot matrix to the
# ---------------------------------
def proj_matrices(femspace: FEMSpace, roswrproblem: ROSWRProblem, pod: POD, T: float, t: float, weighted: bool = False) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    projs = {}
    poderr = {}
    snapshot_matrix = np.zeros((femspace.nnodes, pod.ntime)) # shape (n_interior, n_time)
    snapshot_matrix[femspace.interior_nodes,:] = pod.compute_snapshots()
    for subdomain in roswrproblem.subdomains:
        subsnap = roswrproblem.restrict(snapshot_matrix, subdomain.domainID)[subdomain.interior_nodes(),:] # Restrict the snapshot matrix to the subdomain
        if not weighted:
            V, S, _ = np.linalg.svd(subsnap, full_matrices=False)
            _, dS, _ = np.linalg.svd(subsnap[:, 1:] - subsnap[:, :-1], full_matrices=False)
            projs[subdomain.domainID] = V[:, :pod.r]
            poderr[subdomain.domainID] = np.sqrt(2)*np.exp(T/2)*np.sqrt(t*np.sum(dS[pod.r:]**2)) + np.sqrt(np.sum(S[pod.r:]**2))
        else:
            L = cholesky(roswrproblem.subproblems[subdomain.domainID].stiffness_matrix_II, lower=True)
            V, S, _ = np.linalg.svd(L @ subsnap, full_matrices=False)
            projs[subdomain.domainID] = solve_triangular(L, V[:, :pod.r], lower=True)
            _, dS, _ = np.linalg.svd(L @ (subsnap[:, 1:] - subsnap[:, :-1]), full_matrices=False)
            poderr[subdomain.domainID] = np.sqrt(2)*np.exp(T/2)*np.sqrt(t*np.sum(dS[pod.r:]**2)) + np.sqrt(np.sum(S[pod.r:]**2)) # Compute the POD truncation error for the subdomain as the sum of squares of the discarded singular values
    return projs, poderr

# ---------------------------------
# 1D Example (Full vs Reduced Heat problem)
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

# Compute POD modes for the full-order problem to be used in the Reduced Schwarz method
pod1D = POD(heat_problem = problem1D, ntime = 31, lift = 'nodal', theta = 0.5, r = 10)

# Reduce the 1D Heat problem using a projection matrix
rom1D = ReducedHeatProblem(heat_problem = problem1D, V = pod1D.V)

# Solve the 1D Reduced Heat problem using nodal lifting and theta method with theta = 0.5 (Crank-Nicolson) and reconstruct the full solution
rom_solution1D = rom1D.solve(time_grid = time_grid, lift = 'nodal', theta = 0.5, reconstruct = True)

# Define Schwarz problem with 2 subdomains and overlap of 1 layer of elements with version 1 of the decomposition algorithm
oswrproblem1D = OSWRProblem(femspace = femspace1D, t0 = t0, T = T, f = func1D, g = exact1D, h = h1D, n = 3, overlap = 1, version = 1)

# Define Reduced Schwarz problem with 2 subdomains and overlap of 1 layer of elements with version 1 of the decomposition algorithm
roswrproblem1D = ROSWRProblem(femspace = femspace1D, t0 = t0, T = T, f = func1D, g = exact1D, h = h1D, n = 3, overlap = 1, version = 1)

# Construct projection matrices for each subdomain by restricting the global snapshot matrix to the subdomain and performing SVD to extract the first r modes as projection matrix for the subdomain
projs1D, poderr1D = proj_matrices(femspace1D, roswrproblem1D, pod1D, T, time_grid[40])

# Solve the problem using the Schwarz method with RAS, nodal lifting and theta method with theta = 0.5 (Crank-Nicolson) and store the solution of subdomain 2 at time step 40 for visualization
oswr_solution1D = oswrproblem1D.solve(time_grid = time_grid, theta = 0.5, lift = 'nodal', method = 'RAS', maxiter = 250, tol = 1e-14, store_solution = (2, 40)) 

# Solve the problem using the Reduced Schwarz method with RAS, nodal lifting and theta method with theta = 0.5 (Crank-Nicolson) and store the solution of subdomain 2 at time step 40 for visualization
roswr_solution1D = roswrproblem1D.solve(projs = projs1D, time_grid = time_grid, theta = 0.5, lift = 'nodal', method = 'RAS', maxiter = 250, tol = 1e-14, store_solution = (2, 40))

# Store the solution of the Schwarz method for visualization
oswr_spec1D = np.column_stack(oswrproblem1D.solution)

# Store the solution of the Reduced Schwarz method for visualization
roswr_spec1D = np.column_stack(roswrproblem1D.solution)

# Error analysis (compute L2 error between reduced schwarz solution and exact solution, as well as between reduced schwarz solution and fem solution for each time step and report the maximum error across all time steps)
error1D = np.linalg.norm(roswr_solution1D - exact1D(mesh1D.vertices[:, None], time_grid[None, :]), axis = 0)
error_fem1D = np.linalg.norm(heat_solution1D - roswr_solution1D, axis = 0)
print("max error (reduced schwarz vs exact):", error1D.max())
print("max error (reduced schwarz vs fem):", error_fem1D.max())

# Compute the L2 error between the Schwarz and Reduced Schwarz solutions at time step 40 for second subdomain
iter1D = min(oswr_spec1D.shape[1], roswr_spec1D.shape[1])
print(f"POD truncation error for subdomain 2: {poderr1D[2]:.6e}")
error_spec1D = np.zeros(iter1D)
error_mor_spec1D = np.zeros(iter1D)
for i in range(iter1D):
    error_spec1D[i] = ErrorNorms(femspace = oswrproblem1D.subproblems[2].femspace, u1 = oswr_spec1D[:, i:i+1], u2 = roswr_spec1D[:, i:i+1]).compute('l2')
    error_mor_spec1D[i] = ErrorNorms(femspace = oswrproblem1D.subproblems[2].femspace, u1 = roswr_spec1D[:, i:i+1], u2 = roswrproblem1D.restrict(rom_solution1D[:, 40:41], 2)).compute('l2')

# Visualization
visualizer1D = visualize.SolutionVisualizer(mesh1D, roswr_solution1D)
visualizer1D.plot_convergence(error_history = error_spec1D, linewidth = 0.8, markersize = 3, ylabel = r"$\| u_h - u_{r} \|_{L^2}$")
visualizer1D.plot_convergence(error_history = error_mor_spec1D, linewidth = 0.8, markersize = 3, ylabel = r"$\| u_r - u_{rom} \|_{L^2}$")

# ---------------------------------
# 2D Example (Full vs Reduced Heat problem)
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

# Compute POD modes for the full-order problem to be used in the Reduced Schwarz method
pod2D = POD(heat_problem = problem2D, ntime = 31, lift = 'nodal', theta = 0.5, r = 10)

# Reduce the 2D Heat problem using a projection matrix
rom2D = ReducedHeatProblem(heat_problem = problem2D, V = pod2D.V)

# Solve the 2D Reduced Heat problem using nodal lifting and theta method with theta = 0.5 (Crank-Nicolson) and reconstruct the full solution
rom_solution2D = rom2D.solve(time_grid = time_grid, lift = 'nodal', theta = 0.5, reconstruct = True)

# Define Schwarz problem with 3 subdomains and overlap of 1 layer of elements with version 1 of the decomposition algorithm
oswrproblem2D = OSWRProblem(femspace = femspace2D, t0 = t0, T = T, f = func2D, g = exact2D, h = h2D, n = 3, overlap = 1, version = 1)

# Define Reduced Schwarz problem with 3 subdomains and overlap of 1 layer of elements with version 1 of the decomposition algorithm
roswrproblem2D = ROSWRProblem(femspace = femspace2D, t0 = t0, T = T, f = func2D, g = exact2D, h = h2D, n = 3, overlap = 1, version = 1)

# Construct projection matrices for each subdomain by restricting the global snapshot matrix to the subdomain and performing SVD to extract the first r modes as projection matrix for the subdomain
projs2D, poderr2D = proj_matrices(femspace2D, roswrproblem2D, pod2D, T, time_grid[80])

# Solve the problem using the Schwarz method with RAS, nodal lifting and theta method with theta = 0.5 (Crank-Nicolson) and store the solution of subdomain 2 at time step 80 for visualization
oswr_solution = oswrproblem2D.solve(time_grid = time_grid, theta = 0.5, lift = 'nodal', method = 'RAS', maxiter = 250, tol = 1e-14, store_solution = (2, 80))

# Solve the problem using the Reduced Schwarz method with RAS, nodal lifting and theta method with theta = 0.5 (Crank-Nicolson) and store the solution of subdomain 2 at time step 80 for visualization
roswr_solution2D = roswrproblem2D.solve(projs = projs2D, time_grid = time_grid, theta = 0.5, lift = 'nodal', method = 'RAS', maxiter = 250, tol = 1e-14, store_solution = (2, 80))

# Store the solution of the Schwarz method for visualization
oswr_spec2D = np.column_stack(oswrproblem2D.solution)

# Store the solution of the Reduced Schwarz method for visualization
roswr_spec2D = np.column_stack(roswrproblem2D.solution)

# Error analysis (compute L2 error between reduced schwarz solution and exact solution, as well as between reduced schwarz solution and fem solution for each time step and report the maximum error across all time steps)
error2D = np.linalg.norm(roswr_solution2D - exact2D(femspace2D.mesh.vertices[:,0][:, None], femspace2D.mesh.vertices[:,1][:, None], time_grid[None, :]), axis = 0)
error_fem2D = np.linalg.norm(heat_solution2D - roswr_solution2D, axis = 0)
print("max error (reduced schwarz vs exact):", error2D.max())
print("max error (reduced schwarz vs fem):", error_fem2D.max())

# Compute the L2 error between the Schwarz and Reduced Schwarz solutions at time step 80 for second subdomain
iter2D = min(oswr_spec2D.shape[1], roswr_spec2D.shape[1])
print(f"POD truncation error for subdomain 2: {poderr2D[2]:.6e}")
error_spec2D = np.zeros(iter2D)
error_mor_spec2D = np.zeros(iter2D)
for i in range(iter2D):
    error_spec2D[i] = ErrorNorms(femspace = oswrproblem2D.subproblems[1].femspace, u1 = oswr_spec2D[:, i:i+1], u2 = roswr_spec2D[:, i:i+1]).compute('l2')
    error_mor_spec2D[i] = ErrorNorms(femspace = oswrproblem2D.subproblems[1].femspace, u1 = roswr_spec2D[:, i:i+1], u2 = roswrproblem2D.restrict(rom_solution2D[:, 80:81], 2)).compute('l2')

# Visualization
visualizer2D = visualize.SolutionVisualizer(femspace2D.mesh, roswr_solution2D)
visualizer2D.plot_convergence(error_history = error_spec2D, linewidth = 0.8, markersize = 3, ylabel = r"$\| u_h - u_{r} \|_{L^2}$")
visualizer2D.plot_convergence(error_history = error_mor_spec2D, linewidth = 0.8, markersize = 3, ylabel = r"$\| u_r - u_{rom} \|_{L^2}$")