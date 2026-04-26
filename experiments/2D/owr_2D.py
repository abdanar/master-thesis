import numpy as np
from rich.console import Console
from rich.panel import Panel
import visualization.visualize as visualize
from fem.femspace import FEMSpace
from fem.mesh import Mesh
from fom.heat_fom import HeatProblem
from fom.oswr_heat import OSWRProblem
from utils.errornorms import NormType
from utils.history import HistoryConfig, MetricSpec, MetricType, SpatialMode, TemporalMode
from utils.logger import configure_logging
configure_logging(level="info")

# Create a simple square mesh - space mesh generation
vertices = np.array([[0,0],[1,0],[1,1],[0,1]])
mesh2D = Mesh(vertices = vertices, options = 'qa0.01')

# Print mesh information using rich console
console = Console()
console.print(Panel(mesh2D.info(), title = "[bold yellow]Mesh Information[/bold yellow]", border_style="cyan", expand = False))

# Time domain definition (ntime = (T - t0)/dt + 1 for uniform time steps)
t0, T, ntime = 0.0, 1.0, 101
time_grid = np.linspace(t0, T, ntime)

# Linear Lagrange finite element space
femspace2D = FEMSpace(mesh = mesh2D, degree = 1)

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

# Define Schwarz problem with 2 subdomains and overlap of 1 layer of elements with version 1 of the interface boundary decomposition
oswrproblem2D = OSWRProblem(heat_problem = problem2D, n = 2, overlap = 1, version = 1)

# Define a history configuration
subdomains = [1, 2]
time_indices = np.array([30, 60])
metrics = [MetricSpec(MetricType.ABSOLUTE_ERROR, SpatialMode.GLOBAL, TemporalMode.STATIC),
           MetricSpec(MetricType.CONVERGENCE_RATE, SpatialMode.SUBDOMAINS, TemporalMode.TIME)]
config = HistoryConfig(metrics = metrics, norm = NormType.L2, exact = exact2D, uh = heat_solution2D, time_indices = time_indices, subdomains = subdomains, mode = 'exact')

# Solve the problem using the Schwarz method with RAS, nodal lifting and theta method with theta = 0.5 (Crank-Nicolson)
history2D, oswr_solution2D = oswrproblem2D.solve(time_grid = time_grid, theta = 0.5, lift = 'nodal', method = 'RAS', maxiter = 100, tol = 1e-9, histconfig = config)

# Error analysis (compute L2 error between schwarz solution and exact solution, as well as between schwarz solution and fem solution for each time step and report the maximum error across all time steps)
error2D = np.linalg.norm(oswr_solution2D - exact2D(femspace2D.mesh.vertices[:,0][:, None], femspace2D.mesh.vertices[:,1][:, None], time_grid[None, :]), axis = 0)
error_fem2D = np.linalg.norm(heat_solution2D - oswr_solution2D, axis = 0)
print("max error (schwarz vs exact):", error2D.max())
print("max error (schwarz vs fem):", error_fem2D.max())

# Mesh visualization
visualizer2D = visualize.MeshVisualizer(mesh2D)
visualizer2D.plot_subdomains(subdomains = oswrproblem2D.subdomains, membership = oswrproblem2D.membership, 
                            show_vertex_markers = False, show_node_numbers = False, show_element_numbers = False)

# Solution visualization
global_history = history2D.values[MetricType.ABSOLUTE_ERROR]["global"] # shape (niter,)
subdomain_history = history2D.values[MetricType.CONVERGENCE_RATE]["subdomains"] # shape dictionary of shape {domainID: shape (niter, ntime)}
visualizer2D = visualize.SolutionVisualizer(femspace2D.mesh, oswr_solution2D)
styles = {1: {'color': 'orange', 'linestyle': '-', 'linewidth': 0.8},
          2: {'color': 'blue', 'linestyle': '-', 'linewidth': 0.8}}
visualizer2D.plot_iteration(data = global_history, ylabel = r"$\| u_{exact} - u_{OSWR} \|_{L^2}$", save_path="figures/2D/fig2D_global_fom(exact).png",
                              color = 'black', linestyle = '-', linewidth = 0.8)
for i in range(len(time_indices)):
    visualizer2D.plot_iteration(data = {str(domainID): subdomain_history[domainID][:, i] for domainID in subdomain_history}, title = r"Convergence Rate", xlabel = r"Iteration ($k$)", 
                                  ylabel = rf"$\dfrac{{\| u_{{exact}}(t_{{{time_indices[i]}}}) - u^{{k}}_{{OSWR}}(t_{{{time_indices[i]}}}) \|_{{L^2}}}}{{\| u_{{exact}}(t_{{{time_indices[i]}}}) - u^{{k-1}}_{{OSWR}}(t_{{{time_indices[i]}}}) \|_{{L^2}}}}$",
                                  save_path=f"figures/2D/fig2D_subdomains_time{time_indices[i]}_fom(exact).png", styles = styles)