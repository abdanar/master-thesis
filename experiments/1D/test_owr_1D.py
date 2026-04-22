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

# Create a mesh - space mesh generation
vert1D = np.linspace(0, 1, 100)
mesh1D = Mesh(vertices = vert1D, dim = 1)

# Print mesh information using rich console
console = Console()
console.print(Panel(mesh1D.info(), title = "[bold yellow]Mesh Information[/bold yellow]", border_style="cyan", expand = False))

# Time domain definition (ntime = (T - t0)/dt + 1 for uniform time steps)
t0, T, ntime = 0.0, 1.0, 101
time_grid = np.linspace(t0, T, ntime)

# Linear Lagrange finite element space
femspace1D = FEMSpace(mesh = mesh1D, domain = 'interval', degree = 1)

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

# Define Schwarz problem with 2 subdomains and overlap of 1 layer of elements
oswrproblem1D = OSWRProblem(femspace = femspace1D, t0 = t0, T = T, f = func1D, g = exact1D, h = h1D, n = 2, overlap = 1)

# Define a history configuration
subdomains = [1, 2]
time_indices = np.array([40, 80])
metrics = [MetricSpec(MetricType.ABSOLUTE_ERROR, SpatialMode.GLOBAL, TemporalMode.STATIC),
           MetricSpec(MetricType.CONVERGENCE_RATE, SpatialMode.SUBDOMAINS, TemporalMode.TIME)]
config = HistoryConfig(metrics = metrics, norm = NormType.L2, exact = exact1D, uh = heat_solution1D, time_indices = time_indices, subdomains = subdomains, mode = 'exact')

# Solve the problem using the Schwarz method with RAS, nodal lifting and theta method with theta = 0.5 (Crank-Nicolson)
history1D, oswr_solution1D = oswrproblem1D.solve(time_grid = time_grid, theta = 0.5, lift = 'nodal', method = 'RAS', maxiter = 400, tol = 1e-12, histconfig = config)

# Error analysis (compute L2 error between schwarz solution and exact solution, as well as between schwarz solution and fem solution for each time step and report the maximum error across all time steps)
error1D = np.linalg.norm(oswr_solution1D - exact1D(femspace1D.mesh.vertices[:, None], time_grid[None, :]), axis = 0)
error_fem1D = np.linalg.norm(heat_solution1D - oswr_solution1D, axis = 0)
print("max error (schwarz vs exact):", error1D.max())
print("max error (schwarz vs fem):", error_fem1D.max())

# Visualization
global_history = history1D.values[MetricType.ABSOLUTE_ERROR]["global"] # shape (niter,)
subdomain_history = history1D.values[MetricType.CONVERGENCE_RATE]["subdomains"] # shape dictionary of shape {domainID: shape (niter, ntime)}
visualizer1D = visualize.SolutionVisualizer(femspace1D.mesh, oswr_solution1D)
styles = {1: {'color': 'orange', 'linestyle': '-', 'linewidth': 0.8},
          2: {'color': 'blue', 'linestyle': '-', 'linewidth': 0.8}}
visualizer1D.plot_convergence(error_history = global_history, dpi = 300, ylabel = r"$\| u_{exact} - u_{OSWR} \|_{L^2}$", save_path="figures/1D/fig1D_global_fom(exact).png",
                              color = 'black', linestyle = '-', linewidth = 0.8)
for i in range(len(time_indices)):
    visualizer1D.plot_convergence(error_history = {domainID: subdomain_history[domainID][:, i] for domainID in subdomain_history}, dpi = 300, title = r"Convergence Rate", xlabel = r"Iteration ($k$)", 
                                  ylabel = rf"$\dfrac{{\| u_{{exact}}(t_{{{time_indices[i]}}}) - u^{{k}}_{{OSWR}}(t_{{{time_indices[i]}}}) \|_{{L^2}}}}{{\| u_{{exact}}(t_{{{time_indices[i]}}}) - u^{{k-1}}_{{OSWR}}(t_{{{time_indices[i]}}}) \|_{{L^2}}}}$",
                                  save_path=f"figures/1D/fig1D_subdomains_time{time_indices[i]}_fom(exact).png", styles = styles)