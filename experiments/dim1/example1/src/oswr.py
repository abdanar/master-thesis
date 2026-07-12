import numpy as np
import matplotlib.pyplot as plt
from experiments.dim1.example1.src import setup
from experiments.experiment_utils import plot_solution_and_error, plot_contour_and_error, max_error
from fom.heat_oswr import OSWRHeat
from utils import history
from utils.logger import configure_logging, get_logger
from visualization.parula import parula
from visualization.visualize import plot
configure_logging(level="info")
logger = get_logger(__name__)

# Load necessary data
heat_solution_short = np.load(setup.data_dir/f"fem_T{setup.Tshort}.npy")
heat_solution_long = np.load(setup.data_dir/f"fem_T{setup.Tlong}.npy")

def oswr_heatsolve(heat_solution, heat_problem, decomposition_info, time_grid, **kwargs):
    histconfig = history.HistoryConfig(metrics = [setup.metric], uh = heat_solution, subdomains = decomposition_info.subdomain_ids, mode = 'fem')
    oswr_solver = OSWRHeat(heat_problem, decomposition_info)
    oswr_solution = oswr_solver.solve(time_grid, histconfig = histconfig, **kwargs)
    return oswr_solver

# Solve OSWR problems for all three overlap sizes and store the subdomain solutions for the specified iterations
oswr_short, oswr_long = {}, {}
for ext, decomposition_info in setup.decomposition_infos.items():
    oswr_short[ext] = oswr_heatsolve(heat_solution_short, setup.heat_short, decomposition_info, setup.time_grid_short, lift = 'nodal', theta = 1, maxiter = 180, tol = 1e-12, store_solution = [1, 2, 3, 4])
    oswr_long[ext] = oswr_heatsolve(heat_solution_long, setup.heat_long, decomposition_info, setup.time_grid_long, lift = 'nodal', theta = 1, maxiter = 180, tol = 1e-12, store_solution = [1, 2, 3, 4])

# Plot the subdomain solutions and errors for iterations n = 1, 2, 3, 4 when overlap size is 0.1 (overlap = 5) for both short and long time intervals
fig_short, axes_short = plot_solution_and_error(setup.mesh1D.vertices, setup.time_grid_short, heat_solution_short, oswr_short[5].iterates, oswr_short[5].ltog, 
                                                iterations = [1, 2, 3, 4], plt_kw = {'azim': 255, 'elev': 15, 'zlim': (-1, 1)}, 
                                                suptitle = rf"OSWR Subdomain Solutions and Associated Errors ($T={setup.Tshort}$)")
fig_long, axes_long = plot_solution_and_error(setup.mesh1D.vertices, setup.time_grid_long, heat_solution_long, oswr_long[5].iterates, oswr_long[5].ltog, 
                                              iterations = [1, 2, 3, 4], plt_kw = {'azim': 255, 'elev': 15, 'zlim': (-1.5, 2)}, suptitle = rf"OSWR Subdomain Solutions and Associated Errors ($T={setup.Tlong}$)")

# Plot the contour of the combined OSWR solutions and errors for iterations n = 1, 2, 3, 4 when overlap size is 0.1 (overlap = 5) for both short and long time intervals
fig_short_contour, axes_short_contour = plot_contour_and_error(setup.mesh1D.vertices, setup.time_grid_short, heat_solution_short, oswr_short[5], iterations = [1, 2, 3, 4], 
                            suptitle = rf"OSWR Solutions and Associated Errors ($T={setup.Tshort}$)", fig_kw = {"figsize": (9, 15)}, plt_kw = {'cmap': parula()})
fig_long_contour, axes_long_contour = plot_contour_and_error(setup.mesh1D.vertices, setup.time_grid_long, heat_solution_long, oswr_long[5], iterations = [1, 2, 3, 4], 
                            suptitle = rf"OSWR Solutions and Associated Errors ($T={setup.Tlong}$)", fig_kw = {"figsize": (9, 15)}, plt_kw = {'cmap': parula()})

# EXPERIMENT: Compute maximum error between OSWR and FEM solution for both short and long time intervals for all three overlap sizes
error_short, error_long = {}, {}
markers = ['o', 's', '^']
colors = ["#045B8E", "#ED2939", "#3F9C35"]
fig_iter, axes = plt.subplots(1, 2, dpi = 150, figsize = (10, 4))
for i, ext in enumerate(setup.decomposition_infos.keys()):
    assert oswr_short[ext].history is not None and oswr_long[ext].history is not None, "History object for short or long time interval is None"
    error_short[ext] = max_error(oswr_short[ext].history)
    error_long[ext] = max_error(oswr_long[ext].history)
    plot(np.arange(1, error_short[ext].size + 1), error_short[ext], xlabel = 'Iteration', ylabel = 'Maximum error', title = rf"Maximum error as a function of the iteration ($T={setup.Tshort}$)", 
                   logscale = True, plot_kwargs = {'linewidth': 0.6, 'linestyle': '-','marker': markers[i], 'markerfacecolor': 'none', 'markersize' : 6, 'markeredgewidth': 0.8, 'color': colors[i], 'label': f'{ext*0.02} overlap'}, ax = axes[0], xlim = (0, 30), ylim = (1e-9, 1e-0))
    plot(np.arange(1, error_long[ext].size + 1), error_long[ext], xlabel = 'Iteration', ylabel = 'Maximum error', title = rf"Maximum error as a function of the iteration ($T={setup.Tlong}$)", 
                   logscale = True, plot_kwargs = {'linewidth': 0.6, 'linestyle': '-','marker': markers[i], 'markerfacecolor': 'none', 'markersize' : 4, 'markeredgewidth': 0.8, 'color': colors[i], 'label': f'{ext*0.02} overlap'}, ax = axes[1], xlim = (0, 85), ylim = (1e-9, 1e-0))
legend1 = axes[0].legend(frameon=True, fancybox=False, facecolor="white", edgecolor="black")
legend2 = axes[1].legend(frameon=True, fancybox=False, facecolor="white", edgecolor="black")
legend1.get_frame().set_linewidth(0.7)
legend2.get_frame().set_linewidth(0.7)

# Save the figures
fig_short.savefig(setup.fig_dir/f"oswr/oswr_T{setup.Tshort}.svg", dpi = 300, bbox_inches = 'tight')
fig_long.savefig(setup.fig_dir/f"oswr/oswr_T{setup.Tlong}.svg", dpi = 300, bbox_inches = 'tight')
fig_short_contour.savefig(setup.fig_dir/f"oswr/oswr_T{setup.Tshort}_contour.svg", dpi = 300, bbox_inches = 'tight')
fig_long_contour.savefig(setup.fig_dir/f"oswr/oswr_T{setup.Tlong}_contour.svg", dpi = 300, bbox_inches = 'tight')
fig_iter.savefig(setup.fig_dir/f"oswr/oswr_maxerr.svg", dpi = 300, bbox_inches = 'tight')

plt.show()