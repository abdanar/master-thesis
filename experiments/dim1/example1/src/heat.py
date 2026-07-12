import numpy as np
import matplotlib.pyplot as plt
from experiments.dim1.example1.src import setup
from utils.errornorms import ErrorNorms
from visualization.parula import parula
from visualization.visualize import plot_contour, plot_wireframe

# Solve both 1D Heat problems using nodal lifting and theta method with theta = 1 (Backward Euler)
heat_solution_short = setup.heat_short.solve(time_grid = setup.time_grid_short, lift = 'nodal', theta = 1)
heat_solution_long = setup.heat_long.solve(time_grid = setup.time_grid_long, lift = 'nodal', theta = 1)

# Plot the solution of the heat equation over a short time interval T = 0.1
fig_short, ax_short = plot_wireframe(X = setup.vertices, Y = setup.time_grid_short, Z = heat_solution_short.T, 
                                    xlabel = r'$x$', ylabel = r'$t$', zlabel = r'$u(x,t)$', title = rf'Solution of the heat equation over $T={setup.Tshort}$',
                                    xlim = (0, 1), ylim = (setup.t0, setup.Tshort), zlim = (-1, 1), azim = 255, elev = 15)

# Plot the solution of the heat equation over a long time interval T = 1
fig_long, ax_long = plot_wireframe(X = setup.vertices, Y = setup.time_grid_long, Z = heat_solution_long.T, 
                                    xlabel = r'$x$', ylabel = r'$t$', zlabel = r'$u(x,t)$', title = rf'Solution of the heat equation over $T={setup.Tlong}$',
                                    xlim = (0, 1), ylim = (setup.t0, setup.Tlong), zlim = (-1.5, 2), azim = 255, elev = 15)

# Plot the contour of the solution of the heat equation over a short time interval T = 0.1
fig_short_contour, ax_short_contour = plot_contour(X = setup.vertices, Y = setup.time_grid_short, Z = heat_solution_short.T, 
                                                    xlabel = r'$x$', ylabel = r'$t$', title = rf'Contour plot of the solution over $T={setup.Tshort}$', 
                                                    xlim = (0, 1), ylim = (setup.t0, setup.Tshort), levels = 10, cline = True, plot_kwargs={'cmap': parula()})

# Plot the contour of the solution of the heat equation over a long time interval T = 1
fig_long_contour, ax_long_contour = plot_contour(X = setup.vertices, Y = setup.time_grid_long, Z = heat_solution_long.T, 
                                                    xlabel = r'$x$', ylabel = r'$t$', title = rf'Contour plot of the solution over $T={setup.Tlong}$', 
                                                    xlim = (0, 1), ylim = (setup.t0, setup.Tlong), levels = 10, cline = True, plot_kwargs={'cmap': parula()})

# Compute the LinfL2-norm FEM error
error_short = ErrorNorms(femspace = setup.femspace1D, u1 = heat_solution_short, u_exact = setup.boundary_condition, time = setup.time_grid_short, mode = 'exact').linf_l2_error()
error_long = ErrorNorms(femspace = setup.femspace1D, u1 = heat_solution_long, u_exact = setup.boundary_condition, time = setup.time_grid_long, mode = 'exact').linf_l2_error()
print(f"\033[96mLinfL2 norm FEM error\033[0m for \033[92mT = {setup.Tshort}\033[0m: \033[91m{error_short:.4e}\033[0m")
print(f"\033[96mLinfL2 norm FEM error\033[0m for \033[92mT = {setup.Tlong}\033[0m: \033[91m{error_long:.4e}\033[0m")

# Save the figures
fig_short.savefig(setup.fig_dir/f"fom/fem_T{setup.Tshort}.svg", dpi = 300, bbox_inches = 'tight')
fig_long.savefig(setup.fig_dir/f"fom/fem_T{setup.Tlong}.svg", dpi = 300, bbox_inches = 'tight')
fig_short_contour.savefig(setup.fig_dir/f"fom/fem_T{setup.Tshort}_contour.svg", dpi = 300, bbox_inches = 'tight')
fig_long_contour.savefig(setup.fig_dir/f"fom/fem_T{setup.Tlong}_contour.svg", dpi = 300, bbox_inches = 'tight')

# Save data
np.save(setup.data_dir/f"fem_T{setup.Tshort}.npy", heat_solution_short)
np.save(setup.data_dir/f"fem_T{setup.Tlong}.npy", heat_solution_long)

plt.show()