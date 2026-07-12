import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from experiments.dim1.example1.src import setup
from rom.pod import POD
from rom.heat_rom import ReducedHeatProblem
from utils.errornorms import ErrorNorms
from utils.logger import configure_logging, get_logger
from visualization.parula import parula
from visualization.visualize import plot_contour, plot_wireframe
configure_logging(level="info")
logger = get_logger(__name__)

# Load necessary data
heat_solution_short = np.load(setup.data_dir/f"fem_T{setup.Tshort}.npy")
heat_solution_long = np.load(setup.data_dir/f"fem_T{setup.Tlong}.npy")

# Solve both 1D Heat problems to obtain snapshots for constructing the POD basis
snapshots_short = setup.heat_short.solve(time_grid = np.linspace(setup.t0, setup.Tshort, 16), lift = 'nodal', theta = 1, homogeneous = True)
snapshots_long = setup.heat_long.solve(time_grid = np.linspace(setup.t0, setup.Tlong, 16), lift = 'nodal', theta = 1, homogeneous = True)

# Weight matrix for the L2 inner product (using the mass matrix of the heat problem)
weight_short_l2 = (setup.heat_short.mass_matrix_II).toarray()
weight_long_l2 = (setup.heat_long.mass_matrix_II).toarray()

# Weight matrix for the H^1_0 inner product (using the stiffness matrix of the heat problem)
weight_short_h1 = (setup.heat_short.stiffness_matrix_II).toarray()
weight_long_h1 = (setup.heat_long.stiffness_matrix_II).toarray()

def rheat_solve(snapshots, r, heat_problem, time_grid, weight = None, option = 'noDQ', dt = None, **kwargs):
    pod_reductor = POD(snapshots = snapshots, r = r, weight = weight)
    if option == 'DQ':
        if dt is None:
            raise ValueError("dt must be provided for DQ POD.")
        pod_reductor.snapshots = pod_reductor.dq_snapshots(dt)
    rheat = ReducedHeatProblem(heat_problem = heat_problem, V = pod_reductor.basis())
    return rheat.solve(time_grid = time_grid, weight = weight, **kwargs)

def error_norms(rvalues: list, snapshots, heat_problem, time_grid, heat_solution: np.ndarray, weight = None, option = 'noDQ', dt = None, **kwargs):
    data = np.zeros(len(rvalues))
    for i, r in enumerate(rvalues):
        rheat_solution = rheat_solve(snapshots = snapshots, r = r, heat_problem = heat_problem, time_grid = time_grid, weight = weight, option = option, dt = dt, **kwargs)
        data[i] = ErrorNorms(femspace = setup.femspace1D, u1 = rheat_solution, u2 = heat_solution, time = time_grid, mode = 'fem').linf_l2_error()
    return np.vectorize(lambda x: f"{x:.4e}")(data)

# Solve both 1D Heat problems using nodal lifting and theta method with theta = 1 (Backward Euler)
r = 4
rheat_solution_short = rheat_solve(snapshots = snapshots_short, r = r, heat_problem = setup.heat_short, time_grid = setup.time_grid_short, weight = weight_short_l2, lift = 'nodal', theta = 1)
rheat_solution_long = rheat_solve(snapshots = snapshots_long, r = r, heat_problem = setup.heat_long, time_grid = setup.time_grid_long, weight = weight_long_l2, lift = 'nodal', theta = 1)

# Plot the reduced solution of the heat equation over short and long time intervals
fig_short, ax_short = plot_wireframe(X = setup.mesh1D.vertices, Y = setup.time_grid_short, Z = rheat_solution_short.T,
                                     title = rf'Reduced solution of the heat equation ($r = {r}, T={setup.Tshort}$)', 
                                     xlabel = r'$x$', ylabel = r'$t$', zlabel = r'$u_{r}(x,t)$',
                                     xlim = (0, 1), ylim = (setup.t0, setup.Tshort), zlim = (-1, 1), azim = 255, elev = 15)
fig_long, ax_long = plot_wireframe(X = setup.mesh1D.vertices, Y = setup.time_grid_long, Z = rheat_solution_long.T,
                                   title = rf'Reduced solution of the heat equation ($r = {r}, T={setup.Tlong}$)', 
                                   xlabel = r'$x$', ylabel = r'$t$', zlabel = r'$u_{r}(x,t)$',
                                   xlim = (0, 1), ylim = (setup.t0, setup.Tlong), zlim = (-1.5, 2), azim = 255, elev = 15)

# Plot the error between the reduced solution and the full solution vs time for both short and long time intervals
fig_error_short, ax_error_short = plot_wireframe(X = setup.mesh1D.vertices, Y = setup.time_grid_short, Z = (heat_solution_short - rheat_solution_short).T, 
                                                 title = rf'Error between reduced and the full solution ($r = {r}, T={setup.Tshort}$)', logscale = True, 
                                                 xlabel = r'$x$', ylabel = r'$t$', zlabel = r'$|u_{FEM}(x,t) - u_{r}(x,t)|$', azim = 160, elev = 15,
                                                 xlim = (0, 1), ylim = (setup.t0, setup.Tshort))
fig_error_long, ax_error_long = plot_wireframe(X = setup.mesh1D.vertices, Y = setup.time_grid_long, Z = (heat_solution_long - rheat_solution_long).T, 
                                               title = rf'Error between reduced and the full solution ($r = {r}, T={setup.Tlong}$)', logscale = True, 
                                               xlabel = r'$x$', ylabel = r'$t$', zlabel = r'$|u_{FEM}(x,t) - u_{r}(x,t)|$', azim = 160, elev = 15,
                                               xlim = (0, 1), ylim = (setup.t0, setup.Tlong))

# Plot contour of the error between the reduced solution and the full solution for both short and long time intervals
fig_error_short_contour, ax_error_short_contour = plot_contour(X = setup.mesh1D.vertices, Y = setup.time_grid_short, Z = (heat_solution_short - rheat_solution_short).T, 
                                                               title = rf'Error between reduced and the full solution ($r = {r}, T={setup.Tshort}$)', logscale = True,
                                                               xlabel = r'$x$', ylabel = r'$t$', plot_kwargs={'cmap': parula()})
fig_error_long_contour, ax_error_long_contour = plot_contour(X = setup.mesh1D.vertices, Y = setup.time_grid_long, Z = (heat_solution_long - rheat_solution_long).T, 
                                                             title = rf'Error between reduced and the full solution ($r = {r}, T={setup.Tlong}$)', logscale = True,
                                                             xlabel = r'$x$', ylabel = r'$t$', plot_kwargs={'cmap': parula()})

# Compute the LinfL2 norm errors for both short and long time intervals for different values of r
r_values = [2, 4, 6, 8, 10]
## POD vs FEM solution (T = 1)
err_long_l2 = error_norms(weight = weight_long_l2, rvalues = r_values, snapshots = snapshots_long, heat_problem = setup.heat_long,
                                    time_grid = setup.time_grid_long, heat_solution = heat_solution_long, lift = 'nodal', theta = 1)
err_long_h1 = error_norms(weight = weight_long_h1, rvalues = r_values, snapshots = snapshots_long, heat_problem = setup.heat_long,
                                    time_grid = setup.time_grid_long, heat_solution = heat_solution_long, lift = 'nodal', theta = 1)
## DQ POD vs FEM solution (T = 1)
err_long_dq_l2= error_norms(option = 'DQ',dt = setup.dt_long, weight = weight_long_l2, rvalues = r_values, snapshots = snapshots_long, heat_problem = setup.heat_long,
                                    time_grid = setup.time_grid_long, heat_solution = heat_solution_long, lift = 'nodal', theta = 1)
err_long_dq_h1 = error_norms(option = 'DQ',dt = setup.dt_long, weight = weight_long_h1, rvalues = r_values, snapshots = snapshots_long, heat_problem = setup.heat_long,
                                    time_grid = setup.time_grid_long, heat_solution = heat_solution_long, lift = 'nodal', theta = 1)
## POD vs FEM solution (T = 0.1)
err_short_l2 = error_norms(weight = weight_short_l2, rvalues = r_values, snapshots = snapshots_short, heat_problem = setup.heat_short,
                                    time_grid = setup.time_grid_short, heat_solution = heat_solution_short, lift = 'nodal', theta = 1)
err_short_h1 = error_norms(weight = weight_short_h1, rvalues = r_values, snapshots = snapshots_short, heat_problem = setup.heat_short,
                                    time_grid = setup.time_grid_short, heat_solution = heat_solution_short, lift = 'nodal', theta = 1)
## DQ POD vs FEM solution (T = 0.1)
err_short_dq_l2= error_norms(option = 'DQ',dt = setup.dt_short, weight = weight_short_l2, rvalues = r_values, snapshots = snapshots_short, heat_problem = setup.heat_short,
                                    time_grid = setup.time_grid_short, heat_solution = heat_solution_short, lift = 'nodal', theta = 1)
err_short_dq_h1 = error_norms(option = 'DQ',dt = setup.dt_short, weight = weight_short_h1, rvalues = r_values, snapshots = snapshots_short, heat_problem = setup.heat_short,
                                    time_grid = setup.time_grid_short, heat_solution = heat_solution_short, lift = 'nodal', theta = 1)
print(f"\033[1;36m" f"Table 1. Error comparison of POD and DQ POD reduced-order models " f"(T = {setup.Tshort})" f"\033[0m")
df_short = pd.DataFrame({("POD", r"$X=L^2(0,1)$"): err_short_l2, ("POD", r"$X=H^1_0(0,1)$"): err_short_h1,
                   ("DQ POD", r"$X=L^2(0,1)$"): err_short_dq_l2, ("DQ POD", r"$X=H^1_0(0,1)$"): err_short_dq_h1}, index=r_values)
df_short.index.name = r"$r$"
print(df_short)
print(f"\033[1;36m" f"Table 2. Error comparison of POD and DQ POD reduced-order models " f"(T = {setup.Tlong})" f"\033[0m")
df_long = pd.DataFrame({("POD", r"$X=L^2(0,1)$"): err_long_l2, ("POD", r"$X=H^1_0(0,1)$"): err_long_h1,
                   ("DQ POD", r"$X=L^2(0,1)$"): err_long_dq_l2, ("DQ POD", r"$X=H^1_0(0,1)$"): err_long_dq_h1}, index=r_values)
df_long.index.name = r"$r$"
print(df_long)

# Save the figures
fig_short.savefig(setup.fig_dir/f"rom/rfem_r{r}_T{setup.Tshort}.svg", dpi = 300, bbox_inches = 'tight')
fig_long.savefig(setup.fig_dir/f"rom/rfem_r{r}_T{setup.Tlong}.svg", dpi = 300, bbox_inches = 'tight')
fig_error_short.savefig(setup.fig_dir/f"rom/rfem_pterr_r{r}_T{setup.Tshort}.svg", dpi = 300, bbox_inches = 'tight')
fig_error_long.savefig(setup.fig_dir/f"rom/rfem_pterr_r{r}_T{setup.Tlong}.svg", dpi = 300, bbox_inches = 'tight')
fig_error_short_contour.savefig(setup.fig_dir/f"rom/rfem_pterr_contour_r{r}_T{setup.Tshort}.svg", dpi = 300, bbox_inches = 'tight')
fig_error_long_contour.savefig(setup.fig_dir/f"rom/rfem_pterr_contour_r{r}_T{setup.Tlong}.svg", dpi = 300, bbox_inches = 'tight')

# Save data
np.save(setup.data_dir/f"snapshots_T{setup.Tshort}.npy", snapshots_short)
np.save(setup.data_dir/f"snapshots_T{setup.Tlong}.npy", snapshots_long)

plt.show()