import sys
from typing import Callable, Optional
import numpy as np
from tqdm import trange
from fem.linearsolver import DirectSolver, LinearSolver
from fom.heat import HeatProblem
import utils.logger as log

logger = log.setup_logger(__name__, level = 'info')

class ReducedHeatProblem:
    def __init__(self, heat_problem: HeatProblem, V: np.ndarray):
        """
        Initialize the reduced heat problem by projecting the full-order heat problem onto a reduced basis.

        Parameters:
        -----------
        heat_problem: HeatProblem
            The full-order heat problem containing the FEM space, mass and stiffness matrices, load vector, initial condition, 
            and boundary condition function.
        V: np.ndarray
            The projection matrix for reducing the full-order problem to a lower-dimensional subspace.
        """
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
        self.nintnodes = len(self.interior_nodes)

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

    def evaluate(self, g: Callable, time_grid: np.ndarray):
        if self.femspace.dim == 1:
            return g(self.verts[self.boundary_nodes][:, None], time_grid[None, :])
        elif self.femspace.dim == 2:
            return g(self.verts[self.boundary_nodes][:, 0][:, None], self.verts[self.boundary_nodes][:, 1][:, None], time_grid[None, :])
        else:
            raise ValueError("Unsupported dimension for boundary condition g.")

    def vectorize(self, g: Callable | np.ndarray, time_grid: np.ndarray):
        if isinstance(g, np.ndarray):
            return g
        elif isinstance(g, Callable):
            return self.evaluate(g, time_grid)
        else:
            raise ValueError("Unsupported type for boundary condition g.")

    def solve_nodal(self, time_grid: np.ndarray, theta: float = 0.5, solver: LinearSolver = DirectSolver(), reuse_load: bool = False, g_new: Optional[Callable | np.ndarray] = None, **kwargs):
        """
        Solves the reduced heat problem using a nodal lifting approach for the boundary conditions.

        Parameters:
        -----------
        time_grid: np.ndarray
            Array of time steps at which to solve the heat equation, i.e., [t0, t1, ..., T].
            Uniform time steps are assumed for the time-stepping scheme.
        theta : float
            Parameter for the θ-method time-stepping scheme.
            - θ = 0: Explicit Euler (conditionally stable)
            - θ = 0.5: Crank-Nicolson (unconditionally stable, second-order accurate)
            - θ = 1: Implicit Euler (unconditionally stable, first-order accurate)
            Default is 0.5 (Crank-Nicolson).
        solver : LinearSolver
            Linear solver to use for solving the linear system. Must be an instance of a class that 
            inherits from `LinearSolver`. Default is `DirectSolver()`.
        reuse_load : bool
            If True, the reduced load vectors for all time steps will be computed once and reused for subsequent calls with 
            `reuse_load=True` to this method. This can save computation time if the same reduced load vectors are needed multiple 
            times (e.g., in a Schwarz waveform relaxation context). Default is False. 
        g_new : Callable or np.ndarray, optional
            New boundary condition to use for the simulation. If None, the existing boundary condition 
            `self.g` will be used. This is useful for scenarios where the boundary condition changes 
            between calls to `solve_nodal` and we want to update it without creating a new `ReducedHeatProblem` 
            instance. Default is None.
        **kwargs : dict
            Additional keyword arguments to pass to the linear solver.
        """
        # Total number of time nodes (including initial and final time)
        ntime = len(time_grid)

        # Total number of time steps
        nsteps = ntime - 1

        # Time step size (assuming uniform time steps)
        dt = (self.T - self.t0)/nsteps

        # Update the boundary condition values if a new g is provided (either as a function or as a numpy array)
        g_to_use = g_new if g_new is not None else self.g
        self.dirichlet_values = self.vectorize(g_to_use, time_grid) # shape (nbdnodes, ntime)

        # Project the initial condition onto the reduced space
        icond_r = self.V.T @ self.hp.icond[self.interior_nodes]

        # Pre-allocate solution array: columns are time steps
        solution = np.zeros((self.V.shape[1], ntime)) # shape (r, ntime)

        # Initial reduced solution
        solution[:, 0] = icond_r

        # Initial condition for the interior nodes in the reduced space
        u_interior = icond_r

        logger.debug(f"Using reduced θ-method time-stepping with θ = {theta} | Starting time integration | nsteps = {nsteps}, dt = {dt:.3e}")

        # Precompute constant parts of the system matrix for the reduced θ-method
        lhs_const_r = self.mass_matrix_II_r + theta*dt*self.stiffness_matrix_II_r
        rhs_const_r = self.mass_matrix_II_r - (1 - theta)*dt*self.stiffness_matrix_II_r

        # Precompute the load vectors for all time steps
        if reuse_load and hasattr(self, 'F_all_r') and self.F_all_r.shape[1] == ntime:
            F_all_r = self.F_all_r
            logger.info("Reusing previously computed reduced load vectors for all time steps...")
        else:
            logger.info("Computing reduced load vectors for all time steps...")
            F_all_r = np.column_stack([self.load_vector_I_r(t) for t in time_grid])

        if reuse_load:
            self.F_all_r = F_all_r

        logger.info("Computing boundary contributions for all time steps...")

        # Columns are psi_{n+1} - psi_n for n = 0, ..., nsteps-1
        delta_psi_np1 = self.dirichlet_values[:, 1:] - self.dirichlet_values[:, :-1] # shape (nbdnodes, nsteps)
        delta_psi_n = np.column_stack([delta_psi_np1[:, 0:1], delta_psi_np1[:, :-1]]) # shape (nbdnodes, nsteps)

        # For nodal lifting, the lift values are zero
        self.lift_values = np.zeros((self.nintnodes, ntime)) # shape (nintnodes, ntime)

        # Precompute reduced boundary contributions for all steps (vectorized)
        R_all_r = dt*(theta*F_all_r[:, 1:] + (1-theta)*F_all_r[:, :-1]) - theta * self.mass_matrix_IB_r @ delta_psi_np1 - dt * theta * self.stiffness_matrix_IB_r @ self.dirichlet_values[:, 1:] - (1-theta) * self.mass_matrix_IB_r @ delta_psi_n - dt * (1-theta) * self.stiffness_matrix_IB_r @ self.dirichlet_values[:, :-1]  # shape (r, nsteps)

        logger.info("Starting time-stepping loop with tqdm progress bar...")

        # Time-stepping loop with tqdm
        step = 0
        t = self.t0
        with trange(nsteps, desc = "\033[92mReduced Heat Solver\033[0m", unit="step",ascii = "░▒█", ncols = 100, disable = not sys.stdout.isatty()) as pbar:
            for step in pbar:
                pbar.set_postfix_str(f"\033[93mt={t:.3e}\033[0m")
                t += dt
                rhs_r = rhs_const_r @ u_interior + R_all_r[:, step] # shape (r,)
                u = solver.solve(lhs_const_r, rhs_r, **kwargs) # shape (r,)
                solution[:, step + 1] = u # shape (r,)
                u_interior = u
        return solution

    def solve_harmonic(self, time_grid: np.ndarray, theta: float = 0.5, solver: LinearSolver = DirectSolver(), reuse_load: bool = False, g_new: Optional[Callable | np.ndarray] = None, **kwargs):
        
        # Total number of time nodes (including initial and final time)
        ntime = len(time_grid)

        # Total number of time steps
        nsteps = ntime - 1

        # Time step size (assuming uniform time steps)
        dt = (self.T - self.t0) / nsteps

        # Update the boundary condition values if a new g is provided (either as a function or as a numpy array)
        g_to_use = g_new if g_new is not None else self.g
        self.dirichlet_values = self.vectorize(g_to_use, time_grid) # shape (nbdnodes, ntime)

        # Pre-allocate solution array: columns are time steps
        solution = np.zeros((self.V.shape[1], ntime)) # shape (r, ntime)

        logger.info(f"Using reduced θ-method time-stepping with θ = {theta} | Starting time integration | nsteps = {nsteps}, dt = {dt:.3e}")

        # Precompute constant parts of the system matrix for the reduced θ-method
        lhs_const_r = self.mass_matrix_II_r + theta*dt*self.stiffness_matrix_II_r
        rhs_const_r = self.mass_matrix_II_r - (1 - theta)*dt*self.stiffness_matrix_II_r

        # Precompute the load vectors for all time steps
        if reuse_load and hasattr(self, 'F_all_r') and self.F_all_r.shape[1] == ntime:
            F_all_r = self.F_all_r
            logger.info("Reusing previously computed load vectors for all time steps...")
        else:
            logger.info("Computing load vectors for all time steps...")
            F_all_r = np.column_stack([self.load_vector_I_r(t) for t in time_grid])
            
        if reuse_load:
            self.F_all_r = F_all_r  # store only if reuse is enabled

        logger.info("Computing boundary contributions for all time steps...")

        # Columns are psi_{n+1} - psi_n for n = 0, ..., nsteps-1
        delta_psi_np1 = self.dirichlet_values[:, 1:] - self.dirichlet_values[:, :-1] # shape (nbdnodes, nsteps)
        delta_psi_n = np.column_stack([delta_psi_np1[:, 0:1], delta_psi_np1[:, :-1]]) # shape (nbdnodes, nsteps)

        logger.info("Computing harmonic lifting values for all time steps...")

        # Columns are l_{n+1} - l_n for n = 0, ..., nsteps-1 (computed by solving the harmonic lifting problem for each time step)
        lift_values = np.zeros((self.nintnodes, ntime)) # shape (nintnodes, ntime)
        for i in range(ntime):
            lift_values[:, i] = solver.solve(A = self.stiffness_matrix_II, b = -self.stiffness_matrix_IB @ self.dirichlet_values[:, i], **kwargs) # shape (nintnodes,)
        self.lift_values = lift_values  # Store lift values for potential reuse
        lift_np1 = lift_values[:, 1:] - lift_values[:, :-1] # shape (nintnodes, nsteps)
        lift_n = np.column_stack([lift_np1[:, 0:1], lift_np1[:, :-1]]) # shape (nintnodes, nsteps)

        # Precompute reduced boundary contributions for all steps (vectorized)
        R_all_r = dt*(theta*F_all_r[:, 1:] + (1-theta)*F_all_r[:, :-1]) - theta * self.mass_matrix_IB_r @ delta_psi_np1 - theta * self.mass_matrix_II_r @ lift_np1 - (1-theta) * self.mass_matrix_IB_r @ delta_psi_n - (1-theta) * self.mass_matrix_II_r @ lift_n  # shape (r, nsteps)

        logger.info("Starting time-stepping loop with tqdm progress bar...")

        # Time-stepping loop with tqdm
        step = 0
        t = self.t0
        u_interior = self.V.T @ (self.hp.icond[self.interior_nodes] - lift_values[:, 0]) # shape (r,)
        solution[:, 0] = u_interior
        with trange(nsteps, desc = "\033[92mHeat Solver\033[0m", unit="step",ascii = "░▒█", ncols = 100, disable = not sys.stdout.isatty()) as pbar:
            for step in pbar:
                pbar.set_postfix_str(f"\033[93mt={t:.3e}\033[0m")
                t += dt
                rhs = rhs_const_r @ u_interior + R_all_r[:, step]
                u = solver.solve(lhs_const_r, rhs, **kwargs)
                solution[:, step + 1] = u # shape (r,)
                u_interior = u
        return solution

    def solve(self, time_grid: np.ndarray, lift: str = 'nodal', theta: float = 0.5, solver: LinearSolver = DirectSolver(), reuse_load: bool = False, g_new: Optional[Callable | np.ndarray] = None, reconstruct: bool = True, **kwargs):
        if lift == 'nodal':
            hom_solution_r = self.solve_nodal(time_grid = time_grid, theta = theta, solver = solver, reuse_load = reuse_load, g_new = g_new, **kwargs)
        elif lift == 'harmonic':
            hom_solution_r = self.solve_harmonic(time_grid = time_grid, theta = theta, solver = solver, g_new = g_new, **kwargs)
        # elif lift == 'parabolic':
        #   hom_solution_r = self.solve_parabolic(ntime = ntime, theta = theta, solver = solver, g_new = g_new, **kwargs)
        else:
            raise ValueError(f"Unsupported lift method: {lift}")
        if reconstruct: # Project back to full space
            return self.reconstruct(hom_solution_r, time_grid = time_grid, g_new = g_new)
        else:
            return hom_solution_r
    
    def reconstruct(self, hom_solution_r: np.ndarray, time_grid: np.ndarray, g_new: Optional[Callable | np.ndarray] = None) -> np.ndarray:
        solution = np.zeros((self.nnodes, hom_solution_r.shape[1])) # shape (nnodes, ntime)
        solution[self.interior_nodes, :] = self.V @ hom_solution_r + self.lift_values # shape (nintnodes, ntime)
        solution[self.boundary_nodes, :] = self.vectorize(g_new if g_new is not None else self.g, time_grid) # shape (nbdnodes, ntime)
        solution[:, 0] = self.hp.icond # Ensure initial condition is correctly set in the full space
        return solution