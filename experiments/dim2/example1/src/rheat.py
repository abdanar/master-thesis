import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rom.pod import POD
from rom.heat_rom import ReducedHeatProblem
from experiments.dim2.example1.src import setup
from utils.errornorms import ErrorNorms
from utils.logger import configure_logging, get_logger
configure_logging(level="info")
logger = get_logger(__name__)

# Load necessary data
heat_solution_short = np.load(setup.data_dir/f"fem_T{setup.Tshort}.npy")
heat_solution_long = np.load(setup.data_dir/f"fem_T{setup.Tlong}.npy")

# Solve both 1D Heat problems to obtain snapshots for constructing the POD basis
snapshots_short = setup.heat_short.solve(time_grid = np.linspace(setup.t0, setup.Tshort, 21), lift = 'nodal', theta = 1, homogeneous = True)
snapshots_long = setup.heat_long.solve(time_grid = np.linspace(setup.t0, setup.Tlong, 21), lift = 'nodal', theta = 1, homogeneous = True)

# Weight matrix for the L2 inner product (using the mass matrix of the heat problem)
weight_short_l2 = (setup.heat_short.mass_matrix_II).toarray()
weight_long_l2 = (setup.heat_long.mass_matrix_II).toarray()

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
        data[i] = ErrorNorms(femspace = setup.femspace2D, u1 = rheat_solution, u2 = heat_solution, time = time_grid, mode = 'fem').linf_l2_error()
    return np.vectorize(lambda x: f"{x:.4e}")(data)

# Compute the LinfL2 norm errors for both short and long time intervals for different values of r
r_values = [2, 4, 6, 8, 10]
## L^2-POD vs FEM solution (T = 0.1)
err_short_l2= error_norms(weight = weight_short_l2, rvalues = r_values, snapshots = snapshots_short, heat_problem = setup.heat_short,
                                    time_grid = setup.time_grid_short, heat_solution = heat_solution_short, lift = 'nodal', theta = 1)
## L^2-POD vs FEM solution (T = 3)
err_long_l2 = error_norms(weight = weight_long_l2, rvalues = r_values, snapshots = snapshots_long, heat_problem = setup.heat_long,
                                    time_grid = setup.time_grid_long, heat_solution = heat_solution_long, lift = 'nodal', theta = 1)
print(f"\033[1;36m" f"Table 1. LinfL2 norm errors of L^2-POD reduced-order models \033[0m")
df = pd.DataFrame({f"T = {setup.Tshort}": err_short_l2, f"T = {setup.Tlong}": err_long_l2}, index=r_values)
df.index.name = r"$r$"
print(df)

# Save data
np.save(setup.data_dir/f"snapshots_T{setup.Tshort}.npy", snapshots_short)
np.save(setup.data_dir/f"snapshots_T{setup.Tlong}.npy", snapshots_long)

plt.show()