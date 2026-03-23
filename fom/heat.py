import sys
import numpy as np
from scipy.sparse import coo_array
from tqdm import trange
from fem.boundary import DirichletBC
from fem.femspace import FEMSpace
from fem.assembler import Assembler
from fem.linearsolver import *
import utils.logger as log 
from typing import Callable

logger = log.setup_logger(__name__, level = 'info')

class HeatProblem:
    def __init__(self, femspace: FEMSpace, t0: float, T: float, f: Callable, g: Callable | dict, h: Callable):
        """
        FEM solver for the heat equation:
            - 1D: u_t(x, t) - u_xx(x, t)) = f(x, t),   x ∈ Ω, t ∈ (t0, T)
            - 2D: u_t(x, y, t) - ∇·(∇u(x, y, t)) = f(x, y, t),   (x, y) ∈ Ω, t ∈ (t0, T)
        with Dirichlet boundary conditions 
            - 1D: u(x, t) = g(x, t) for x on the boundary of Ω and t ∈ (t0, T)
            - 2D: u(x, y, t) = g(x, y, t) for (x, y) on the boundary of Ω and t ∈ (t0, T)
        and initial condition
                - 1D: u(x, t0) = h(x)
                - 2D: u(x, y, t0) = h(x, y)

        Parameters
        ----------
        femspace : FEMSpace
            FEM space object
        t0 : float
            Initial time
        T : float
            Final time
        f : Callable
            Source term f(x, t) for 1D or f(x, y, t) for 2D
        g : Callable or dict
            The Dirichlet boundary condition function or dictionary.
            - If it is a function, it should be defined as g(x, t) for 1D or g(x, y, t) for 2D problems.
            - If it is a dictionary, the keys should be global node indices corresponding to the boundary nodes, 
              and the values should be numpy arrays of length equal to the number of time steps, containing the 
              Dirichlet values at those nodes for each time step. For example: {0: [0.0, 0.45, 3.4], 5: [1.0, 1.0, 1.0], ...}
        h : Callable
            The initial condition function h(x) for 1D or h(x, y) for 2D
        """
        self.femspace = femspace
        self.t0 = t0
        self.T = T
        self.f = f
        self.g = g
        self.dim = femspace.dim
        verts = self.femspace.mesh.vertices
        self.assembler = Assembler(self.femspace)
        self.mass_matrix, self.stiffness_matrix = self._assemble_space()
        if self.dim == 1:
            self.F = lambda t: self.assembler.global_load_vector(lambda x: self.f(x, t))
            self.h = h(verts) # shape (n_vertices,)
        else:
            self.F = lambda t: self.assembler.global_load_vector(lambda x, y: self.f(x, y, t))
            self.h = h(verts[:,0], verts[:,1]) # shape (n_vertices,)
            
    def _assemble_space(self) -> tuple[coo_array, coo_array]:
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

    def solve_nodal(self, ntime: int, theta: float = 0.5, solver: LinearSolver = DirectSolver(), **kwargs) -> np.ndarray:

        # Total number of time steps
        nsteps = ntime - 1

        # Generate time steps (assuming uniform time steps)
        time_steps = np.linspace(self.t0, self.T, ntime)

        # Time step size (assuming uniform time steps)
        dt = (self.T - self.t0) / nsteps

        # Pre-allocate solution array: columns are time steps
        solution = np.zeros((self.mass_matrix.shape[0], ntime))

        # Initial solution
        solution[:, 0] = self.h

        logger.debug(f"Using θ-method time-stepping with θ = {theta} | Starting time integration | nsteps = {nsteps}, dt = {dt:.3e}")

        # Precompute constant parts of the system matrix for the θ-method
        lhs_const = self.mass_matrix + theta*dt*self.stiffness_matrix
        rhs_const = self.mass_matrix - (1 - theta)*dt*self.stiffness_matrix

        # Apply boundary conditions to the system matrix (for nodal lifting, we will modify the system at each time step)
        boundary = DirichletBC(femspace = self.femspace, g = self.g, time_steps = time_steps)

        # Apply boundary conditions to the constant part of the system matrix (this will modify the matrix structure for nodal lifting)
        lhs, _ = boundary.apply(K = lhs_const, modify_K = True)

        # Time-stepping loop with tqdm
        step = 0
        t = self.t0
        u_initial = self.h[:, None]  # shape (n_vertices, 1)
        load_initial = self.F(t)
        with trange(nsteps, desc = "\033[92mHeat Solver\033[0m", unit="step",ascii = "░▒█", ncols = 100, disable = not sys.stdout.isatty()) as pbar:
            for step in pbar:
                pbar.set_postfix_str(f"\033[93mt={t:.3e}\033[0m")
                t += dt
                load_vector = self.F(t)
                rhs = rhs_const @ u_initial + dt*(theta*load_vector + (1 - theta)*load_initial)
                _, rhs = boundary.apply(K = lhs_const, rhs = rhs, time_step = step, modify_K = False)
                u = solver.solve(lhs, rhs, **kwargs)
                solution[:, step + 1] = u.ravel()
                u_initial = u
                load_initial = load_vector
                step += 1
        return solution

    def solve(self, ntime: int, lift: str = 'nodal', theta: float = 0.5, solver: LinearSolver = DirectSolver(), **kwargs) -> np.ndarray:
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
        **kwargs
            Additional keyword arguments to pass to the boundary condition application method 
            (e.g., solver parameters for iterative solvers).

        Returns
        -------
        np.ndarray
            The computed solution vector at the FEM nodes.
        """
        if lift == 'nodal':
            solution = self.solve_nodal(ntime = ntime, theta = theta, solver = solver, **kwargs)
        # elif lift == 'harmonic':
        #     solution = self.solve_harmonic(ntime = ntime, theta = theta, solver = solver, **kwargs)
        # elif lift == 'parabolic':
        #     solution = self.solve_parabolic(ntime = ntime, theta = theta, solver = solver, **kwargs)
        else:
            raise ValueError(f"Unsupported lift method: {lift}")
        return solution
    
# dirichlet = {k: v[step] for k, v in self.dirichlet.items()}
# lhs, rhs = self.assembler.apply_Dirichlet_bc(lhs, rhs, dirichlet)