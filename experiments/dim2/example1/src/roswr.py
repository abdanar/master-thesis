import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rom.pod import POD
from rom.heat_roswr import ROSWRHeat
from utils.errornorms import ErrorNorms
from utils.logger import configure_logging, get_logger
from experiments.dim2.example1.src import setup
from experiments.experiment_utils import plot_contour_and_error_2D
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

def roswr_heatsolve(snapshots, r, heat_problem, decomposition_info, time_grid, weight = None, **kwargs):
    pod_reductor = POD(snapshots = snapshots, r = r, weight = weight)
    roswr_solver = ROSWRHeat(heat_problem, decomposition_info)
    roswr_solution = roswr_solver.solve(pod = pod_reductor, time_grid = time_grid, **kwargs)
    return roswr_solver

# EXPERIMENT 1: Solve ROSWR problems for both short and long time intervals with r = `rorder` and option = 'noDQ' (L2 POD) only for 5 iterations
rorder = 4
roswr_short = roswr_heatsolve(snapshots_short, rorder, setup.heat_short, setup.decomposition_info, setup.time_grid_short, weight = weight_short_l2, lift = 'nodal', theta = 1, option = 'noDQ', maxiter = 5, store_solution = [1, 2, 3, 4])
roswr_long = roswr_heatsolve(snapshots_long, rorder, setup.heat_long, setup.decomposition_info, setup.time_grid_long, weight = weight_long_l2, lift = 'nodal', theta = 1, option = 'noDQ', maxiter = 5, store_solution = [1, 2, 3, 4])

## Plot the contour of the combined ROSWR solutions and errors for iterations n = 1, 2, 3, 4 at final time for both short and long time intervals
fig_short_contour, axes_short_contour = plot_contour_and_error_2D(setup.mesh2D.vertices[:,0], setup.mesh2D.vertices[:,1], heat_solution_short, roswr_short, iterations = [1, 2, 3, 4], 
                            suptitle = rf"ROSWR Solutions and Associated Errors ($r={rorder}, T={setup.Tshort}$)", fig_kw = {"figsize": (9, 15)}, t = -1)
fig_long_contour, axes_long_contour = plot_contour_and_error_2D(setup.mesh2D.vertices[:,0], setup.mesh2D.vertices[:,1], heat_solution_long, roswr_long, iterations = [1, 2, 3, 4], 
                            suptitle = rf"ROSWR Solutions and Associated Errors ($r={rorder}, T={setup.Tlong}$)", fig_kw = {"figsize": (9, 15)}, t = -1)

# EXPERIMENT 2: Compute maximum LinfL2 norm error between ROSWR and FEM solution for different r values
r_values = [2, 4, 6, 8, 10]
dec_info = setup.decomposition_info
error_data_short, error_data_long = np.zeros(len(r_values)), np.zeros(len(r_values))
for i, r in enumerate(r_values):
    pod_reductor_short = POD(snapshots = snapshots_short, r = r, weight = weight_short_l2)
    pod_reductor_long = POD(snapshots = snapshots_long, r = r, weight = weight_long_l2)
    roswr_solver_short = ROSWRHeat(setup.heat_short, dec_info)
    roswr_solver_long = ROSWRHeat(setup.heat_long, dec_info)
    rdata_short = roswr_solver_short.solve(pod = pod_reductor_short, time_grid = setup.time_grid_short, lift = 'nodal', theta = 1, maxiter = 300, tol = 1e-14, option = 'noDQ', combine = False)
    rdata_long = roswr_solver_long.solve(pod = pod_reductor_long, time_grid = setup.time_grid_long, lift = 'nodal', theta = 1, maxiter = 600, tol = 1e-14, option = 'noDQ', combine = False)
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
fig_short_contour.savefig(setup.fig_dir/f"roswr/roswr_r_{rorder}_T{setup.Tshort}_contour.svg", dpi = 300, bbox_inches = 'tight')
fig_long_contour.savefig(setup.fig_dir/f"roswr/roswr_r_{rorder}_T{setup.Tlong}_contour.svg", dpi = 300, bbox_inches = 'tight')

plt.show()