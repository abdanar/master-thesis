import numpy as np
import matplotlib.pyplot as plt
from experiments.dim2.example1.src import setup
from utils.errornorms import ErrorNorms
from visualization.visualize import plot_tri

# Solve both 2D Heat problems using nodal lifting and theta method with theta = 1 (Backward Euler)
heat_solution_short = setup.heat_short.solve(time_grid = setup.time_grid_short, lift = 'nodal', theta = 1)
heat_solution_long = setup.heat_long.solve(time_grid = setup.time_grid_long, lift = 'nodal', theta = 1)

# Plot the solution of the heat equation over a short time interval T = 0.1
fig_short, ax_short = plot_tri(x = setup.mesh2D.vertices[:,0], y = setup.mesh2D.vertices[:,1], z = heat_solution_short[:, -1], 
                                xlabel = r'$x$', ylabel = r'$y$', contour = True, cline = True, levels = 10, 
                                title = rf'Solution of the heat equation at $T={setup.Tshort}$', xlim = (0, 1), ylim = (0, 1))

# Plot the solution of the heat equation over a long time interval T = 1
fig_long, ax_long = plot_tri(x = setup.mesh2D.vertices[:,0], y = setup.mesh2D.vertices[:,1], z = heat_solution_long[:, -1], 
                                xlabel = r'$x$', ylabel = r'$y$', contour = True, cline = True, levels = 10,
                                title = rf'Solution of the heat equation at $T={setup.Tlong}$', xlim = (0, 1), ylim = (0, 1))

# Compute the LinfL2-norm FEM error
error_short = ErrorNorms(femspace = setup.femspace2D, u1 = heat_solution_short, u_exact = setup.boundary_condition, time = setup.time_grid_short, mode = 'exact').linf_l2_error()
error_long = ErrorNorms(femspace = setup.femspace2D, u1 = heat_solution_long, u_exact = setup.boundary_condition, time = setup.time_grid_long, mode = 'exact').linf_l2_error()
print(f"\033[96mLinfL2 norm FEM error\033[0m for \033[92mT = {setup.Tshort}\033[0m: \033[91m{error_short:.4e}\033[0m")
print(f"\033[96mLinfL2 norm FEM error\033[0m for \033[92mT = {setup.Tlong}\033[0m: \033[91m{error_long:.4e}\033[0m")

# Save the figures
fig_short.savefig(setup.fig_dir/f"fom/fem_T{setup.Tshort}.svg", dpi = 300, bbox_inches = 'tight')
fig_long.savefig(setup.fig_dir/f"fom/fem_T{setup.Tlong}.svg", dpi = 300, bbox_inches = 'tight')

# Save data
np.save(setup.data_dir/f"fem_T{setup.Tshort}.npy", heat_solution_short)
np.save(setup.data_dir/f"fem_T{setup.Tlong}.npy", heat_solution_long)

plt.show()