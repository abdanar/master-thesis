import sys
import numpy as np
from tqdm import trange
from fem.linearsolver import LinearSolver, DirectSolver
import utils.logger as log 
from typing import Callable, Optional
from fom.heat import HeatProblem

logger = log.setup_logger(__name__, level = 'info')

class ReducedHeatProblem:
    def __init__(self, heat_problem: HeatProblem, V: np.ndarray):

        # Store the full-order heat problem and its parameters
        self.hp = heat_problem
        self.femspace = heat_problem.femspace
        self.g = heat_problem.g
        self.t0 = heat_problem.t0
        self.T = heat_problem.T
        self.verts = self.femspace.mesh.vertices
        self.nnodes = self.femspace.nnodes
        self.boundary_nodes = self.femspace.boundary_nodes
        self.interior_nodes = self.femspace.interior_nodes

        # Check that the projection matrix V has the correct dimensions
        assert V.shape[0] == len(self.interior_nodes), "The number of rows in the projection matrix V must match the number of interior nodes in the FEM space."
        assert V.shape[1] <= V.shape[0], "The number of columns in the projection matrix V must be less than or equal to the number of rows (i.e., n_reduced <= n_interior) for a valid reduction."

        # Store the projection matrix
        self.V = V

        # Extract the relevant submatrices for the interior nodes and the coupling with boundary nodes
        self.mass_matrix_II = self.hp.mass_matrix_II
        self.mass_matrix_IB = self.hp.mass_matrix_IB
        self.stiffness_matrix_II = self.hp.stiffness_matrix_II
        self.stiffness_matrix_IB = self.hp.stiffness_matrix_IB

        # Precompute the reduced mass and stiffness matrices, load vector, and initial condition for the reduced solver
        self.mass_matrix_II_r = V.T @ self.mass_matrix_II @ V
        self.stiffness_matrix_II_r = V.T @ self.stiffness_matrix_II @ V
        self.mass_matrix_IB_r = V.T @ self.mass_matrix_IB
        self.stiffness_matrix_IB_r = V.T @ self.stiffness_matrix_IB
        self.load_vector_I_r = lambda t: V.T @ self.hp.load_vector(t)[self.interior_nodes]
        self.icond_r = V.T @ self.hp.icond[self.interior_nodes]

    def evaluate(self, g: Callable, time_steps: np.ndarray):
        if self.femspace.dim == 1:
            return g(self.verts[self.boundary_nodes][:, None], time_steps[None, :])
        elif self.femspace.dim == 2:
            return g(self.verts[self.boundary_nodes][:, 0][:, None], self.verts[self.boundary_nodes][:, 1][:, None], time_steps[None, :])
        else:
            raise ValueError("Unsupported dimension for boundary condition g.")

    def vectorize(self, g: Callable | np.ndarray, time_steps: np.ndarray):
        if isinstance(g, np.ndarray):
            return g
        elif isinstance(g, Callable):
            return self.evaluate(g, time_steps)
        else:
            raise ValueError("Unsupported type for boundary condition g.")

    def solve_nodal(self, ntime: int, theta: float = 0.5, solver: LinearSolver = DirectSolver(), reuse_load: bool = False, g_new: Optional[Callable | np.ndarray] = None, **kwargs):

        # Total number of time steps
        nsteps = ntime - 1

        # Generate time steps (assuming uniform time steps)
        time_steps = np.linspace(self.t0, self.T, ntime)

        # Time step size (assuming uniform time steps)
        dt = (self.T - self.t0)/nsteps

        # Update the boundary condition values if a new g is provided (either as a function or as a numpy array)
        g_to_use = g_new if g_new is not None else self.g
        self.dirichlet_values = self.vectorize(g_to_use, time_steps) # shape (n_boundary, n_time)

        # Pre-allocate solution array: columns are time steps
        solution = np.zeros((self.V.shape[1], ntime)) # shape (n_reduced, n_time)

        # Initial reduced solution
        solution[:, 0] = self.icond_r

        # Initial condition for the interior nodes in the reduced space
        u_interior = self.icond_r

        logger.debug(f"Using reduced θ-method time-stepping with θ = {theta} | Starting time integration | nsteps = {nsteps}, dt = {dt:.3e}")

        # Precompute constant parts of the system matrix for the θ-method
        lhs_const_r = self.mass_matrix_II_r + theta*dt*self.stiffness_matrix_II_r
        rhs_const_r = self.mass_matrix_II_r - (1 - theta)*dt*self.stiffness_matrix_II_r

        # Precompute the load vectors for all time steps if reuse_load is True, otherwise compute them on the fly in the time-stepping loop
        if reuse_load and hasattr(self, 'F_all_r') and self.F_all_r.shape[1] == ntime:
            F_all_r = self.F_all_r
            logger.info("Reusing previously computed reduced load vectors for all time steps...")
        else:
            logger.info("Computing reduced load vectors for all time steps...")
            F_all_r = np.column_stack([self.load_vector_I_r(t) for t in time_steps])

        if reuse_load:
            self.F_all_r = F_all_r

        logger.info("Computing boundary contributions for all time steps...")

        # Columns are psi_{n+1} - psi_n for n = 0, ..., nsteps-1
        delta_psi_np1 = self.dirichlet_values[:, 1:] - self.dirichlet_values[:, :-1] # shape (n_boundary, n_steps)
        delta_psi_n = np.column_stack([delta_psi_np1[:, 0:1], delta_psi_np1[:, :-1]]) # shape (n_boundary, n_steps)

        # Precompute boundary contributions for all steps (vectorized)
        R_all_r = dt*(theta*F_all_r[:, 1:] + (1-theta)*F_all_r[:, :-1]) - theta * self.mass_matrix_IB_r @ delta_psi_np1 - dt * theta * self.stiffness_matrix_IB_r @ self.dirichlet_values[:, 1:] - (1-theta) * self.mass_matrix_IB_r @ delta_psi_n - dt * (1-theta) * self.stiffness_matrix_IB_r @ self.dirichlet_values[:, :-1]  # shape (n_interior, n_steps)

        logger.info("Starting time-stepping loop with tqdm progress bar...")

        # Time-stepping loop with tqdm
        step = 0
        t = self.t0
        with trange(nsteps, desc = "\033[92mReduced Heat Solver\033[0m", unit="step",ascii = "░▒█", ncols = 100, disable = not sys.stdout.isatty()) as pbar:
            for step in pbar:
                pbar.set_postfix_str(f"\033[93mt={t:.3e}\033[0m")
                t += dt
                rhs_r = rhs_const_r @ u_interior + R_all_r[:, step] # shape (n_reduced,)
                u = solver.solve(lhs_const_r, rhs_r, **kwargs) # shape (n_reduced,)
                solution[:, step + 1] = u # shape (n_reduced,)
                u_interior = u
        return solution

    def solve(self, ntime: int, lift: str = 'nodal', theta: float = 0.5, solver: LinearSolver = DirectSolver(), reuse_load: bool = False, g_new: Optional[Callable | np.ndarray] = None, reconstruct: bool = True, **kwargs):
        if lift == 'nodal':
            solution_r = self.solve_nodal(ntime = ntime, theta = theta, solver = solver, reuse_load = reuse_load, g_new = g_new, **kwargs)
        # elif lift == 'harmonic':
        #     solution_r = self.solve_harmonic(ntime = ntime, theta = theta, solver = solver, g_new = g_new, **kwargs)
        # elif lift == 'parabolic':
        #     solution_r = self.solve_parabolic(ntime = ntime, theta = theta, solver = solver, g_new = g_new, **kwargs)
        else:
            raise ValueError(f"Unsupported lift method: {lift}")
        if reconstruct: # Project back to full space
            solution = np.zeros((self.nnodes, ntime)) # shape (n_nodes, n_time)
            solution[self.interior_nodes, :] = self.V @ solution_r # shape (n_interior, n_time)
            solution[self.boundary_nodes, :] = self.vectorize(g_new if g_new is not None else self.g, np.linspace(self.t0, self.T, ntime)) # shape (n_boundary, n_time)
            return solution
        else:
            return solution_r
    
    def reconstruct(self, solution_r: np.ndarray) -> np.ndarray:
        solution = np.zeros((self.nnodes, solution_r.shape[1])) # shape (n_nodes, n_time)
        solution[self.interior_nodes, :] = self.V @ solution_r # shape (n_interior, n_time)
        solution[self.boundary_nodes, :] = self.vectorize(self.g, np.linspace(self.t0, self.T, solution_r.shape[1])) # shape (n_boundary, n_time)
        return solution