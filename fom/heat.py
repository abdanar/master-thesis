import sys
from typing import Callable, Optional, Literal
import numpy as np
from scipy.sparse import coo_array
from tqdm import trange
from fem.assembler import Assembler
from fem.femspace import FEMSpace
from fem.linearsolver import DirectSolver, LinearSolver
import utils.logger as log

logger = log.setup_logger(__name__, level = 'info')

class HeatProblem:
    def __init__(self, femspace: FEMSpace, t0: float, T: float, f: Callable, g: Callable | np.ndarray, h: Callable):
        """
        Initializes the following heat problem:

            ∂u/∂t - Δu = f          in Ω x (t0, T),
               u(x, t) = g(x, t)    on ∂Ω x (t0, T),
              u(x, t0) = h(x)       in Ω.
        
        Parameters
        ----------
        femspace : FEMSpace
            The finite element space defining the mesh and basis functions.
        t0 : float
            Initial time.
        T : float
            Final time.
        f : Callable
            Source term function. Should be defined as f(x, t) for 1D or f(x, y, t) for 2D problems.
        g : Callable or np.ndarray
            Dirichlet boundary condition. Can be either a function defined as g(x, t) for 1D or g(x, y, t) for 2D problems, 
            or a numpy array of shape (n_boundary_nodes, n_time_steps) containing the boundary values at each time step.
            If numpy array is provided, for each time step n, the boundary values should be given in the order corresponding 
            to the boundary nodes in `femspace.boundary_nodes`.
        h : Callable
            Initial condition function. Should be defined as h(x) for 1D or h(x, y) for 2D problems.
        """
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
        mass : coo_array, shape (nnodes, nnodes)
            Mass matrix
        stiffness : coo_array, shape (nnodes, nnodes)
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
        mass = self.assembler.global_reaction_matrix(reaction = reaction)
        stiffness = self.assembler.global_stiffness_matrix(diffusion = diffusion)
        return mass, stiffness

    def evaluate(self, g: Callable, time_grid: np.ndarray) -> np.ndarray:
        """
        Evaluates the boundary condition function g at the boundary nodes and given time steps.

        Parameters
        ----------
        g : Callable
            Boundary condition function. Should be defined as g(x, t) for 1D or g(x, y, t) for 2D problems.
        time_grid : np.ndarray
            Array of time steps at which to evaluate the boundary condition, shape (ntime,).
        
        Returns
        -------
        np.ndarray
            Evaluated boundary condition values at the boundary nodes and time steps, shape (nbdnodes, ntime).
        """
        if self.femspace.dim == 1:
            return g(self.verts[self.boundary_nodes][:, None], time_grid[None, :])
        elif self.femspace.dim == 2:
            return g(self.verts[self.boundary_nodes][:, 0][:, None], self.verts[self.boundary_nodes][:, 1][:, None], time_grid[None, :])
        else:
            raise ValueError("Unsupported dimension for boundary condition g.")

    def vectorize(self, g: Callable | np.ndarray, time_grid: np.ndarray) -> np.ndarray:
        """
        Converts the boundary condition g into a vectorized form if it is a function, or 
        returns it directly if it is already a numpy array.

        Parameters
        ----------
        g : Callable or np.ndarray
            Boundary condition, either as a function or as a pre-evaluated numpy array.
        time_grid : np.ndarray
            Array of time steps at which to evaluate the boundary condition if g is a function, shape (ntime,).
        
        Returns
        -------
        np.ndarray
            Boundary condition values at the boundary nodes and time steps, shape (nbdnodes, ntime).
        """
        if isinstance(g, np.ndarray):
            return g
        elif isinstance(g, Callable):
            return self.evaluate(g, time_grid)
        else:
            raise ValueError("Unsupported type for boundary condition g.")

    def solve_nodal(self, time_grid: np.ndarray, theta: float = 0.5, solver: LinearSolver = DirectSolver(), reuse_load: bool = False, g_new: Optional[Callable | np.ndarray] = None, homogeneous: bool = False, **kwargs) -> np.ndarray:
        """
        Solves the heat equation using the θ-method time-stepping scheme and directly 
        enforcing Dirichlet boundary conditions at the specified boundary nodes.

        This method corresponds to the "nodal lifting" approach for handling Dirichlet boundary 
        conditions, where the system is modified to directly enforce the boundary values at the 
        specified nodes. 

        Parameters
        ----------
        time_grid : np.ndarray, shape (ntime,)
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
            If True, the load vectors for all time steps will be computed once and reused for subsequent calls with 
            reuse_load=True to this method. This can save computation time if the same load vectors are needed multiple 
            times (e.g., in a Schwarz waveform relaxation context). Default is False. 
        g_new : Callable or np.ndarray, optional
            New boundary condition to use for the simulation. If None, the existing boundary condition 
            `self.g` will be used. This is useful for scenarios where the boundary condition changes 
            between calls to `solve_nodal` and we want to update it without creating a new `HeatProblem` 
            instance. Default is None.
        homogeneous : bool, optional
            If True, the solution returned will only include the values at the interior nodes (homogeneous solution), 
            excluding the boundary nodes. If False, the solution will include values at all nodes (interior + boundary). 
            Default is False.
        **kwargs : dict
            Additional keyword arguments to pass to the linear solver.

        Returns
        -------
        np.ndarray
            Solution array at all time steps, shape (nnodes, ntime) or (nintnodes, ntime) if homogeneous=True.
        """
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
        solution = np.zeros((self.femspace.nnodes, ntime))

        # Initial solution
        solution[:, 0] = self.icond

        # Extract the initial condition values at the interior nodes for the first time step
        u_interior = self.icond[self.interior_nodes] # shape (nintnodes,)

        logger.info(f"Using θ-method time-stepping with θ = {theta} | Starting time integration | nsteps = {nsteps}, dt = {dt:.3e}")

        # Precompute constant parts of the system matrix for the θ-method
        lhs_const = self.mass_matrix_II + theta*dt*self.stiffness_matrix_II
        rhs_const = self.mass_matrix_II - (1 - theta)*dt*self.stiffness_matrix_II

        # Precompute the load vectors for all time steps
        if reuse_load and hasattr(self, 'F_all') and self.F_all.shape[1] == ntime:
            F_all = self.F_all
            logger.info("Reusing previously computed load vectors for all time steps...")
        else:
            logger.info("Computing load vectors for all time steps...")
            F_all = np.column_stack([self.load_vector(t)[self.interior_nodes] for t in time_grid])
            
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
        return solution[self.interior_nodes, :] if homogeneous else solution

    def solve_harmonic(self, time_grid: np.ndarray, theta: float = 0.5, solver: LinearSolver = DirectSolver(), reuse_load: bool = False, g_new: Optional[Callable | np.ndarray] = None, homogeneous: bool = False, **kwargs) -> np.ndarray:
        """
        Solves the heat equation using the θ-method time-stepping scheme and handling Dirichlet boundary conditions
        through a harmonic lifting approach.

        In the harmonic lifting approach, a lifting function is computed at each time step by solving a harmonic problem
        with the given Dirichlet boundary conditions, i.e.,

                -Δl = 0          in Ω,
            l(x, t) = g(x, t)    on ∂Ω.

        The solution is then decomposed into a homogeneous part (which satisfies homogeneous Dirichlet conditions) 
        and a particular part (the lifting function). This allows for more accurate handling of time-dependent 
        boundary conditions and can lead to improved stability and accuracy compared to the nodal lifting approach.

        Parameters
        ----------
        time_grid : np.ndarray, shape (ntime,)
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
            If True, the load vectors for all time steps will be computed once and reused for subsequent calls with 
            reuse_load=True to this method. This can save computation time if the same load vectors are needed multiple 
            times (e.g., in a Schwarz waveform relaxation context). Default is False. 
        g_new : Callable or np.ndarray, optional
            New boundary condition to use for the simulation. If None, the existing boundary condition 
            `self.g` will be used. This is useful for scenarios where the boundary condition changes 
            between calls to solve_nodal and we want to update it without creating a new HeatProblem 
            instance. Default is None.
        homogeneous : bool, optional
            If True, the solution returned will only include the values at the interior nodes (homogeneous solution),
            excluding the boundary nodes. If False, the solution will include values at all nodes (interior + boundary).
            Default is False.
        **kwargs : dict
            Additional keyword arguments to pass to the linear solver.

        Returns
        -------
        np.ndarray
            Solution array at all time steps, shape (nnodes, ntime) or (nintnodes, ntime) if homogeneous=True.
        """
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
        hom_solution = np.zeros((self.nintnodes, ntime))

        logger.info(f"Using θ-method time-stepping with θ = {theta} | Starting time integration | nsteps = {nsteps}, dt = {dt:.3e}")

        # Precompute constant parts of the system matrix for the θ-method
        lhs_const = self.mass_matrix_II + theta*dt*self.stiffness_matrix_II
        rhs_const = self.mass_matrix_II - (1 - theta)*dt*self.stiffness_matrix_II

        # Precompute the load vectors for all time steps
        if reuse_load and hasattr(self, 'F_all') and self.F_all.shape[1] == ntime:
            F_all = self.F_all
            logger.info("Reusing previously computed load vectors for all time steps...")
        else:
            logger.info("Computing load vectors for all time steps...")
            F_all = np.column_stack([self.load_vector(t)[self.interior_nodes] for t in time_grid])
            
        if reuse_load:
            self.F_all = F_all  # store only if reuse is enabled

        logger.info("Computing boundary contributions for all time steps...")

        # Columns are psi_{n+1} - psi_n for n = 0, ..., nsteps-1
        delta_psi_np1 = self.dirichlet_values[:, 1:] - self.dirichlet_values[:, :-1] # shape (nbdnodes, nsteps)
        delta_psi_n = np.column_stack([delta_psi_np1[:, 0:1], delta_psi_np1[:, :-1]]) # shape (nbdnodes, nsteps)

        logger.info("Computing harmonic lifting values for all time steps...")

        # Columns are l_{n+1} - l_n for n = 0, ..., nsteps-1 (computed by solving the harmonic lifting problem for each time step)
        lift_values = np.zeros((self.nintnodes, ntime)) # shape (nintnodes, ntime)
        for i in range(ntime):
            lift_values[:, i] = solver.solve(A = self.stiffness_matrix_II, b = -self.stiffness_matrix_IB @ self.dirichlet_values[:, i], **kwargs) # shape (nintnodes,)
        lift_np1 = lift_values[:, 1:] - lift_values[:, :-1] # shape (nintnodes, nsteps)
        lift_n = np.column_stack([lift_np1[:, 0:1], lift_np1[:, :-1]]) # shape (nintnodes, nsteps)

        # Precompute boundary contributions for all steps (vectorized)
        R_all = dt*(theta*F_all[:, 1:] + (1-theta)*F_all[:, :-1]) - theta * self.mass_matrix_IB @ delta_psi_np1 - theta * self.mass_matrix_II @ lift_np1 - (1-theta) * self.mass_matrix_IB @ delta_psi_n - (1-theta) * self.mass_matrix_II @ lift_n  # shape (nintnodes, nsteps)

        logger.info("Starting time-stepping loop with tqdm progress bar...")

        # Time-stepping loop with tqdm
        step = 0
        t = self.t0
        u_interior = self.icond[self.interior_nodes] - lift_values[:, 0] # shape (nintnodes,)
        hom_solution[:, 0] = u_interior
        with trange(nsteps, desc = "\033[92mHeat Solver\033[0m", unit="step",ascii = "░▒█", ncols = 100, disable = not sys.stdout.isatty()) as pbar:
            for step in pbar:
                pbar.set_postfix_str(f"\033[93mt={t:.3e}\033[0m")
                t += dt
                rhs = rhs_const @ u_interior + R_all[:, step]
                u = solver.solve(lhs_const, rhs, **kwargs)
                hom_solution[:, step + 1] = u # shape (nintnodes,)
                u_interior = u
        if homogeneous: 
            return hom_solution
        solution = np.zeros((self.nnodes, ntime)) # shape (nnodes, ntime)
        solution[self.interior_nodes, :] = hom_solution + lift_values # shape (nintnodes, ntime)
        solution[self.boundary_nodes, :] = self.dirichlet_values # shape (nbdnodes, ntime)
        solution[:, 0] = self.icond # Ensure initial condition is correctly set in the full space
        return solution
    
    def solve_parabolic(self, time_grid: np.ndarray, theta: float = 0.5, solver: LinearSolver = DirectSolver(), reuse_load: bool = False, g_new: Optional[Callable | np.ndarray] = None, **kwargs):

        # Total number of time nodes (including initial and final time)
        ntime = len(time_grid)

        # Total number of time steps
        nsteps = ntime - 1

        # Time step size (assuming uniform time steps)
        dt = (self.T - self.t0) / nsteps

        # Update the boundary condition values if a new g is provided (either as a function or as a numpy array)
        g_to_use = g_new if g_new is not None else self.g
        self.dirichlet_values = self.vectorize(g_to_use, time_grid) # shape (n_boundary, n_time)

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
            F_all = np.column_stack([self.load_vector(t)[self.interior_nodes] for t in time_grid])
            
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
    
    def solve(self, time_grid: np.ndarray, lift: Literal['nodal', 'harmonic', 'parabolic'] = 'nodal', theta: float = 0.5, solver: LinearSolver = DirectSolver(), reuse_load: bool = False, g_new: Optional[Callable | np.ndarray] = None, homogeneous: bool = False, **kwargs):
        """
        Solves the heat equation using the specified time-stepping method and boundary condition handling.

        Parameters
        ----------
        time_grid : np.ndarray, shape (ntime,)
            Array of time steps at which to solve the heat equation, i.e., [t0, t1, ..., T].
            Uniform time steps are assumed for the time-stepping scheme.
        lift : Literal['nodal', 'harmonic', 'parabolic']
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
        reuse_load : bool
            If True, the load vectors for all time steps will be computed once and reused for subsequent
            calls with reuse_load=True to this method. This can save computation time if the same load 
            vectors are needed multiple times (e.g., in a Schwarz waveform relaxation context). Default is False.
        g_new : Callable or np.ndarray, optional
            New boundary condition to use for the simulation. If None, the existing boundary condition 
            `self.g` will be used. This is useful for scenarios where the boundary condition changes 
            between calls to solve and we want to update it without creating a new HeatProblem 
            instance. Default is None.
        homogeneous : bool, optional
            If True, the solution returned will only include the values at the interior nodes (homogeneous solution), 
            excluding the boundary nodes. If False, the solution will include values at all nodes (interior + boundary). 
            Default is False.
        **kwargs
            Additional keyword arguments to pass to the linear solver.

        Returns
        -------
        np.ndarray
            Solution array at all time steps, shape (nnodes, ntime) or (nintnodes, ntime) if homogeneous=True.
        """
        if lift == 'nodal':
            solution = self.solve_nodal(time_grid = time_grid, theta = theta, solver = solver, reuse_load = reuse_load, g_new = g_new, homogeneous = homogeneous, **kwargs)
        elif lift == 'harmonic':
            solution = self.solve_harmonic(time_grid = time_grid, theta = theta, solver = solver, reuse_load = reuse_load, g_new = g_new, homogeneous = homogeneous, **kwargs)
        # elif lift == 'parabolic':
        #     solution = self.solve_parabolic(time_grid = time_grid, theta = theta, solver = solver, g_new = g_new, **kwargs)
        else:
            raise ValueError(f"Unsupported lift method: {lift}. Supported options are 'nodal', 'harmonic', and 'parabolic'.")
        return solution