import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rom.pod import POD
from rom.heat_roswr import ROSWRHeat
from visualization.visualize import plot
import utils.history as history
from utils.errornorms import ErrorNorms
from utils.logger import configure_logging, get_logger
from experiments.dim1.example1.src import setup
from experiments.experiment_utils import max_error
configure_logging(level="info")
logger = get_logger(__name__)

# Each EXPERIMENT can be run independently, to run a specific experiment, comment out the other experiments and run the script.

# Load necessary data
heat_solution_short = np.load(setup.data_dir/f"fem_T{setup.Tshort}.npy")
heat_solution_long = np.load(setup.data_dir/f"fem_T{setup.Tlong}.npy")
snapshots_short = np.load(setup.data_dir/f"snapshots_T{setup.Tshort}.npy")
snapshots_long = np.load(setup.data_dir/f"snapshots_T{setup.Tlong}.npy")

# Weight matrix for the L2 inner product (using the mass matrix of the heat problem)
weight_short_l2 = (setup.heat_short.mass_matrix).toarray()
weight_long_l2 = (setup.heat_long.mass_matrix).toarray()

def roswr_heatsolve(snapshots, r, heat_solution, heat_problem, decomposition_info, time_grid, metrics, weight = None, **kwargs):
    histconfig = history.HistoryConfig(metrics = metrics, uh = heat_solution, subdomains = decomposition_info.subdomain_ids, mode = 'fem')
    pod_reductor = POD(snapshots = snapshots, r = r, weight = weight)
    roswr_solver = ROSWRHeat(heat_problem, decomposition_info)
    roswr_solution = roswr_solver.solve(pod = pod_reductor, time_grid = time_grid, histconfig = histconfig, **kwargs)
    return roswr_solver

# EXPERIMENT 1: Compute maximum Linf error between ROSWR and FEM solution
## Solve ROSWR problems for both short and long time intervals for all three overlap sizes with r = `rorder` and option = 'noDQ' (L2 POD)
rorder = 4
roswr_short, roswr_long = {}, {}
for ext, decomposition_info in setup.decomposition_infos.items():
    roswr_short[ext] = roswr_heatsolve(snapshots_short, rorder, heat_solution_short, setup.heat_short, decomposition_info, setup.time_grid_short, metrics = [setup.metric], weight = weight_short_l2, lift = 'nodal', theta = 1, option = 'noDQ', maxiter = 100, tol = 1e-12, store_solution = [1, 2, 3, 4])
    roswr_long[ext] = roswr_heatsolve(snapshots_long, rorder, heat_solution_long, setup.heat_long, decomposition_info, setup.time_grid_long, metrics = [setup.metric], weight = weight_long_l2, lift = 'nodal', theta = 1, option = 'noDQ', maxiter = 350, tol = 1e-12, store_solution = [1, 2, 3, 4])
## Plot the maximum error as a function of the iteration for both short and long time intervals
error_short, error_long = {}, {}
colors = ["#045B8E", "#ED2939", "#3F9C35"]
fig_iter, axes_iter = plt.subplots(1, 2, dpi = 150, figsize = (10, 4))
for i, ext in enumerate(setup.decomposition_infos.keys()):
    assert roswr_short[ext].history is not None and roswr_long[ext].history is not None, "History object for short or long time interval is None"
    error_short[ext] = max_error(roswr_short[ext].history)
    error_long[ext] = max_error(roswr_long[ext].history)
    plot(np.arange(1, error_short[ext].size + 1), error_short[ext], xlabel = 'Iteration', ylabel = 'Maximum error', title = rf"Maximum error as a function of the iteration ($r = {rorder}, T={setup.Tshort}$)", 
                   logscale = True, plot_kwargs = {'linewidth': 0.7, 'color': colors[i], 'label': f'{ext*0.02} overlap'}, ax = axes_iter[0], xlim = (0, 100))
    plot(np.arange(1, error_long[ext].size + 1), error_long[ext], xlabel = 'Iteration', ylabel = 'Maximum error', title = rf"Maximum error as a function of the iteration ($r = {rorder}, T={setup.Tlong}$)", 
                   logscale = True, plot_kwargs = {'linewidth': 0.7, 'color': colors[i], 'label': f'{ext*0.02} overlap'}, ax = axes_iter[1], xlim = (0, 350))
axes_iter[0].legend(frameon=True, fancybox=False, facecolor="white", edgecolor="black")
axes_iter[1].legend(frameon=True, fancybox=False, facecolor="white", edgecolor="black")

# EXPERIMENT 2: Compute maximum LinfL2 norm error between ROSWR and FEM solution for different r values with overlap size of 0.1
r_values = [2, 4, 6, 8, 10]
dec_info = setup.decomposition_infos[5]
error_data_short, error_data_long = np.zeros(len(r_values)), np.zeros(len(r_values))
for i, r in enumerate(r_values):
    pod_reductor_short = POD(snapshots = snapshots_short, r = r, weight = weight_short_l2)
    pod_reductor_long = POD(snapshots = snapshots_long, r = r, weight = weight_long_l2)
    roswr_solver_short = ROSWRHeat(setup.heat_short, dec_info)
    roswr_solver_long = ROSWRHeat(setup.heat_long, dec_info)
    rdata_short = roswr_solver_short.solve(pod = pod_reductor_short, time_grid = setup.time_grid_short, lift = 'nodal', theta = 1, maxiter = 300, tol = 1e-12, option = 'noDQ', combine = False)
    rdata_long = roswr_solver_long.solve(pod = pod_reductor_long, time_grid = setup.time_grid_long, lift = 'nodal', theta = 1, maxiter = 600, tol = 1e-12, option = 'noDQ', combine = False)
    error_data_short[i], error_data_long[i] = 0, 0
    for id, rsoln in rdata_short.items(): # type: ignore
        error_data_short[i] = max(error_data_short[i], ErrorNorms(femspace = roswr_solver_short.subfemspaces[id], u1 = rsoln, u2 = heat_solution_short[dec_info.ltog[id]], time = setup.time_grid_short, mode = 'fem').linf_l2_error())
    for id, rsoln in rdata_long.items(): # type: ignore
        error_data_long[i] = max(error_data_long[i], ErrorNorms(femspace = roswr_solver_long.subfemspaces[id], u1 = rsoln, u2 = heat_solution_long[dec_info.ltog[id]], time = setup.time_grid_long, mode = 'fem').linf_l2_error())
print(f"\033[1;36m" f"Table 1. LinfL2 norm errors\033[0m")
df = pd.DataFrame({f"T = {setup.Tshort}": np.vectorize(lambda x: f"{x:.4e}")(error_data_short), f"T = {setup.Tlong}": np.vectorize(lambda x: f"{x:.4e}")(error_data_long)}, index=r_values)
df.index.name = r"$r$"
print(df)

# Save the figures
fig_iter.savefig(setup.fig_dir/f"roswr/roswr_maxerr_r{rorder}.svg", dpi = 300, bbox_inches = 'tight')

plt.show()