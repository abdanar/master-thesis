import sys
import numpy as np
from scipy.sparse import coo_array
from tqdm import trange
from fem.boundary import DirichletBC
from fem.femspace import FEMSpace
from fem.assembler import Assembler
from fem.linearsolver import LinearSolver, DirectSolver
import utils.logger as log 
from typing import Callable, Optional

logger = log.setup_logger(__name__, level = 'info')

class HeatProblem:
    def __init__(self, femspace: FEMSpace, t0: float, T: float, f: Callable, g: Callable | np.ndarray, h: Callable):
        
        self.femspace = femspace
        self.t0 = t0
        self.T = T
        self.f = f
        self.g = g
        self.h = h
        self.dim = femspace.dim

        # Extract mesh information from the FEM space for use in the solver
        self.verts = self.femspace.mesh.vertices
        self.nnodes = self.femspace.nnodes
        self.boundary_nodes = self.femspace.boundary_nodes
        self.interior_nodes = self.femspace.interior_nodes
        self.nintnodes = len(self.femspace.interior_nodes)

        # Assemble the global mass and stiffness matrices for the heat equation
        self.assembler = Assembler(self.femspace)
        self.mass_matrix, self.stiffness_matrix = self._assemble_matrices()

        # convert to CSR
        self.mass_matrix = self.mass_matrix.tocsr()
        self.stiffness_matrix = self.stiffness_matrix.tocsr()

        # Extract the relevant submatrices for the interior nodes and the coupling with boundary nodes
        self.mass_matrix_II = self.mass_matrix[np.ix_(self.interior_nodes, self.interior_nodes)]
        self.mass_matrix_IB = self.mass_matrix[np.ix_(self.interior_nodes, self.boundary_nodes)]
        self.stiffness_matrix_II = self.stiffness_matrix[np.ix_(self.interior_nodes, self.interior_nodes)]
        self.stiffness_matrix_IB = self.stiffness_matrix[np.ix_(self.interior_nodes, self.boundary_nodes)]

        # Evaluate the initial condition at the FEM nodes
        if self.dim == 1:
            self.load_vector = lambda t: self.assembler.global_load_vector(lambda x: self.f(x, t))
            self.icond = self.h(self.verts) # shape (n_vertices,)
        elif self.dim == 2:
            self.load_vector = lambda t: self.assembler.global_load_vector(lambda x, y: self.f(x, y, t))
            self.icond = self.h(self.verts[:,0], self.verts[:,1]) # shape (n_vertices,)
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}") 

    def _assemble_matrices(self) -> tuple[coo_array, coo_array]:
        """
        Assemble the global mass and stiffness matrices for the heat equation.

        Returns
        -------
        mass : coo_array
            Mass matrix
        stiffness : coo_array
            Stiffness matrix
        """
        if self.dim == 1:
            diffusion = lambda x: 1
            reaction  = lambda x: 1
        elif self.dim == 2:
            diffusion = lambda x, y: np.eye(2)
            reaction  = lambda x, y: 1
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}")
        mass = self.assembler.global_mass_matrix(reaction = reaction)
        stiffness = self.assembler.global_stiffness_matrix(diffusion = diffusion)
        return mass, stiffness

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
        dt = (self.T - self.t0) / nsteps

        # Update the boundary condition values if a new g is provided (either as a function or as a numpy array)
        g_to_use = g_new if g_new is not None else self.g
        self.dirichlet_values = self.vectorize(g_to_use, time_steps) # shape (n_boundary, n_time)

        # Pre-allocate solution array: columns are time steps
        solution = np.zeros((self.femspace.nnodes, ntime))

        # Initial solution
        solution[:, 0] = self.icond

        # Extract the initial condition values at the interior nodes for the first time step
        u_interior = self.icond[self.interior_nodes] # shape (n_interior,)

        logger.info(f"Using θ-method time-stepping with θ = {theta} | Starting time integration | nsteps = {nsteps}, dt = {dt:.3e}")

        # Precompute constant parts of the system matrix for the θ-method
        lhs_const = self.mass_matrix_II + theta*dt*self.stiffness_matrix_II
        rhs_const = self.mass_matrix_II - (1 - theta)*dt*self.stiffness_matrix_II

        # Precompute the load vectors for all time steps if reuse_load is True, otherwise compute them on the fly in the time-stepping loop
        if reuse_load and hasattr(self, 'F_all') and self.F_all.shape[1] == ntime:
            F_all = self.F_all
            logger.info("Reusing previously computed load vectors for all time steps...")
        else:
            logger.info("Computing load vectors for all time steps...")
            F_all = np.column_stack([self.load_vector(t)[self.interior_nodes] for t in time_steps])
            
        if reuse_load:
            self.F_all = F_all  # store only if reuse is enabled

        logger.info("Computing boundary contributions for all time steps...")

        # Columns are psi_{n+1} - psi_n for n = 0, ..., nsteps-1
        delta_psi_np1 = self.dirichlet_values[:, 1:] - self.dirichlet_values[:, :-1] # shape (n_boundary, n_steps)
        delta_psi_n = np.column_stack([delta_psi_np1[:, 0:1], delta_psi_np1[:, :-1]]) # shape (n_boundary, n_steps)

        # Precompute boundary contributions for all steps (vectorized)
        R_all = dt*(theta*F_all[:, 1:] + (1-theta)*F_all[:, :-1]) - theta * self.mass_matrix_IB @ delta_psi_np1 - dt * theta * self.stiffness_matrix_IB @ self.dirichlet_values[:, 1:] - (1-theta) * self.mass_matrix_IB @ delta_psi_n - dt * (1-theta) * self.stiffness_matrix_IB @ self.dirichlet_values[:, :-1]  # shape (n_interior, n_steps)

        logger.info("Starting time-stepping loop with tqdm progress bar...")

        # Time-stepping loop with tqdm
        step = 0
        t = self.t0
        with trange(nsteps, desc = "\033[92mHeat Solver\033[0m", unit="step",ascii = "░▒█", ncols = 100, disable = not sys.stdout.isatty()) as pbar:
            for step in pbar:
                pbar.set_postfix_str(f"\033[93mt={t:.3e}\033[0m")
                t += dt
                rhs = rhs_const @ u_interior + R_all[:, step]
                u = solver.solve(lhs_const, rhs, **kwargs)
                solution[self.interior_nodes, step + 1] = u # shape (n_interior,)
                u_interior = u
        solution[self.boundary_nodes, :] = self.dirichlet_values
        return solution

    def solve_harmonic(self, ntime: int, theta: float = 0.5, solver: LinearSolver = DirectSolver(), reuse_load: bool = False, g_new: Optional[Callable | np.ndarray] = None, **kwargs):

        # Total number of time steps
        nsteps = ntime - 1

        # Generate time steps (assuming uniform time steps)
        time_steps = np.linspace(self.t0, self.T, ntime)

        # Time step size (assuming uniform time steps)
        dt = (self.T - self.t0) / nsteps

        # Update the boundary condition values if a new g is provided (either as a function or as a numpy array)
        g_to_use = g_new if g_new is not None else self.g
        self.dirichlet_values = self.vectorize(g_to_use, time_steps) # shape (n_boundary, n_time)

        # Pre-allocate solution array: columns are time steps
        solution = np.zeros((self.femspace.nnodes, ntime))

        logger.info(f"Using θ-method time-stepping with θ = {theta} | Starting time integration | nsteps = {nsteps}, dt = {dt:.3e}")

        # Precompute constant parts of the system matrix for the θ-method
        lhs_const = self.mass_matrix_II + theta*dt*self.stiffness_matrix_II
        rhs_const = self.mass_matrix_II - (1 - theta)*dt*self.stiffness_matrix_II

        # Precompute the load vectors for all time steps if reuse_load is True, otherwise compute them on the fly in the time-stepping loop
        if reuse_load and hasattr(self, 'F_all') and self.F_all.shape[1] == ntime:
            F_all = self.F_all
            logger.info("Reusing previously computed load vectors for all time steps...")
        else:
            logger.info("Computing load vectors for all time steps...")
            F_all = np.column_stack([self.load_vector(t)[self.interior_nodes] for t in time_steps])
            
        if reuse_load:
            self.F_all = F_all  # store only if reuse is enabled

        logger.info("Computing boundary contributions for all time steps...")

        # Columns are psi_{n+1} - psi_n for n = 0, ..., nsteps-1
        delta_psi_np1 = self.dirichlet_values[:, 1:] - self.dirichlet_values[:, :-1] # shape (n_boundary, n_steps)
        delta_psi_n = np.column_stack([delta_psi_np1[:, 0:1], delta_psi_np1[:, :-1]]) # shape (n_boundary, n_steps)

        logger.info("Computing harmonic lifting values for all time steps...")

        # Columns are l_{n+1} - l_n for n = 0, ..., nsteps-1 (computed by solving the harmonic lifting problem for each time step)
        lift_values = np.zeros((self.nintnodes, ntime)) # shape (n_interior, n_time)
        for i in range(ntime):
            lift_values[:, i] = solver.solve(A = self.stiffness_matrix_II, b = -self.stiffness_matrix_IB @ self.dirichlet_values[:, i], **kwargs) # shape (n_interior,)
        lift_np1 = lift_values[:, 1:] - lift_values[:, :-1] # shape (n_interior, n_steps)
        lift_n = np.column_stack([lift_np1[:, 0:1], lift_np1[:, :-1]]) # shape (n_interior, n_steps)

        # Precompute boundary contributions for all steps (vectorized)
        R_all = dt*(theta*F_all[:, 1:] + (1-theta)*F_all[:, :-1]) - theta * self.mass_matrix_IB @ delta_psi_np1 - theta * self.mass_matrix_II @ lift_np1 - (1-theta) * self.mass_matrix_IB @ delta_psi_n - (1-theta) * self.mass_matrix_II @ lift_n  # shape (n_interior, n_steps)

        logger.info("Starting time-stepping loop with tqdm progress bar...")

        # Time-stepping loop with tqdm
        step = 0
        t = self.t0
        u_interior = self.icond[self.interior_nodes] - lift_values[:, 0] # shape (n_interior,)
        with trange(nsteps, desc = "\033[92mHeat Solver\033[0m", unit="step",ascii = "░▒█", ncols = 100, disable = not sys.stdout.isatty()) as pbar:
            for step in pbar:
                pbar.set_postfix_str(f"\033[93mt={t:.3e}\033[0m")
                t += dt
                rhs = rhs_const @ u_interior + R_all[:, step]
                u = solver.solve(lhs_const, rhs, **kwargs)
                solution[self.interior_nodes, step + 1] = u # shape (n_interior,)
                u_interior = u
        solution[self.boundary_nodes, :] = self.dirichlet_values
        solution[self.interior_nodes, :] += lift_values
        solution[:, 0] = self.icond
        return solution
    
    def solve_parabolic(self, ntime: int, theta: float = 0.5, solver: LinearSolver = DirectSolver(), reuse_load: bool = False, g_new: Optional[Callable | np.ndarray] = None, **kwargs):

        # Total number of time steps
        nsteps = ntime - 1

        # Generate time steps (assuming uniform time steps)
        time_steps = np.linspace(self.t0, self.T, ntime)

        # Time step size (assuming uniform time steps)
        dt = (self.T - self.t0) / nsteps

        # Update the boundary condition values if a new g is provided (either as a function or as a numpy array)
        g_to_use = g_new if g_new is not None else self.g
        self.dirichlet_values = self.vectorize(g_to_use, time_steps) # shape (n_boundary, n_time)

        # Pre-allocate solution array: columns are time steps
        solution = np.zeros((self.femspace.nnodes, ntime))

        logger.info(f"Using θ-method time-stepping with θ = {theta} | Starting time integration | nsteps = {nsteps}, dt = {dt:.3e}")

        # Precompute constant parts of the system matrix for the θ-method
        lhs_const = self.mass_matrix_II + theta*dt*self.stiffness_matrix_II
        rhs_const = self.mass_matrix_II - (1 - theta)*dt*self.stiffness_matrix_II

        # Precompute the load vectors for all time steps if reuse_load is True, otherwise compute them on the fly in the time-stepping loop
        if reuse_load and hasattr(self, 'F_all') and self.F_all.shape[1] == ntime:
            F_all = self.F_all
            logger.info("Reusing previously computed load vectors for all time steps...")
        else:
            logger.info("Computing load vectors for all time steps...")
            F_all = np.column_stack([self.load_vector(t)[self.interior_nodes] for t in time_steps])
            
        if reuse_load:
            self.F_all = F_all  # store only if reuse is enabled

        logger.info("Computing boundary contributions for all time steps...")

        # Columns are psi_{n+1} - psi_n for n = 0, ..., nsteps-1
        delta_psi_np1 = self.dirichlet_values[:, 1:] - self.dirichlet_values[:, :-1] # shape (n_boundary, n_steps)
        delta_psi_n = np.column_stack([delta_psi_np1[:, 0:1], delta_psi_np1[:, :-1]]) # shape (n_boundary, n_steps)

        logger.info("Computing parabolic lifting values for all time steps...")

        # Parabolic lifting values for all time steps
        lift_values = np.zeros((self.nintnodes, ntime)) # shape (n_interior, n_time)
        for i in range(ntime):
            lift_values[:, i] = solver.solve(A = self.stiffness_matrix_II, b = -self.stiffness_matrix_IB @ self.dirichlet_values[:, i], **kwargs) # shape (n_interior,)
        lift_np1 = lift_values[:, 1:] - lift_values[:, :-1] # shape (n_interior, n_steps)
        lift_n = np.column_stack([lift_np1[:, 0:1], lift_np1[:, :-1]]) # shape (n_interior, n_steps)

        # Precompute boundary contributions for all steps (vectorized)
        R_all = dt*(theta*F_all[:, 1:] + (1-theta)*F_all[:, :-1]) - theta * self.mass_matrix_IB @ delta_psi_np1 - theta * self.mass_matrix_II @ lift_np1 - (1-theta) * self.mass_matrix_IB @ delta_psi_n - (1-theta) * self.mass_matrix_II @ lift_n  # shape (n_interior, n_steps)

        logger.info("Starting time-stepping loop with tqdm progress bar...")

        # Time-stepping loop with tqdm
        step = 0
        t = self.t0
        u_interior = self.icond[self.interior_nodes] - lift_values[:, 0] # shape (n_interior,)
        with trange(nsteps, desc = "\033[92mHeat Solver\033[0m", unit="step",ascii = "░▒█", ncols = 100, disable = not sys.stdout.isatty()) as pbar:
            for step in pbar:
                pbar.set_postfix_str(f"\033[93mt={t:.3e}\033[0m")
                t += dt
                rhs = rhs_const @ u_interior + R_all[:, step]
                u = solver.solve(lhs_const, rhs, **kwargs)
                solution[self.interior_nodes, step + 1] = u # shape (n_interior,)
                u_interior = u
        solution[self.boundary_nodes, :] = self.dirichlet_values
        solution[self.interior_nodes, :] += lift_values
        solution[:, 0] = self.icond
        return solution
    
    def solve(self, ntime: int, lift: str = 'nodal', theta: float = 0.5, solver: LinearSolver = DirectSolver(), reuse_load: bool = False, g_new: Optional[Callable | np.ndarray] = None,**kwargs):
        """
        Solves the heat equation using the specified time-stepping method and boundary condition handling.

        Parameters
        ----------
        lift : str
            Method for handling Dirichlet boundary conditions. Options are:
                - 'nodal': Directly modify the system to enforce Dirichlet BCs at the specified nodes.
                - 'harmonic': Compute a harmonic lifting function that satisfies the Dirichlet BCs and 
                   solve for the homogeneous part of the solution.
                - 'parabolic': Compute a parabolic lifting function that satisfies the Dirichlet BCs and 
                   solve for the homogeneous part of the solution.
            Default is 'nodal'.
        theta : float
            Parameter for the θ-method time-stepping scheme.
            - θ = 0: Explicit Euler (conditionally stable)
            - θ = 0.5: Crank-Nicolson (unconditionally stable, second-order accurate)
            - θ = 1: Implicit Euler (unconditionally stable, first-order accurate)
            Default is 0.5 (Crank-Nicolson).
        solver : LinearSolver
            Linear solver to use for solving the linear system. Must be an instance of a class that 
            inherits from `LinearSolver`. Default is `DirectSolver()`.
        g_new : Callable or dict, optional
            New Dirichlet boundary condition function or dictionary to update the boundary handler before solving.
            - If it is a function, it should be defined as g_new(x) for 1D or g_new(x, y) for 2D problems.
            - If it is a dictionary, the keys should be global node indices corresponding to the boundary nodes, 
              and the values should be the new Dirichlet values at those nodes. For example: {0: 0.0, 5: 1.0, ...}
            If `g_new` is None, the existing boundary conditions defined by `self.g` will be used without modification.
        **kwargs
            Additional keyword arguments to pass to the boundary condition application method 
            (e.g., solver parameters for iterative solvers).

        Returns
        -------
        np.ndarray
            The computed solution vector at the FEM nodes.
        """
        if lift == 'nodal':
            solution = self.solve_nodal(ntime = ntime, theta = theta, solver = solver, reuse_load = reuse_load, g_new = g_new, **kwargs)
        elif lift == 'harmonic':
            solution = self.solve_harmonic(ntime = ntime, theta = theta, solver = solver, reuse_load = reuse_load, g_new = g_new, **kwargs)
        # elif lift == 'parabolic':
        #     solution = self.solve_parabolic(ntime = ntime, theta = theta, solver = solver, g_new = g_new, **kwargs)
        else:
            raise ValueError(f"Unsupported lift method: {lift}")
        return solution