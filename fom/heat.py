import sys
import numpy as np
from scipy.sparse import coo_array
from tqdm import trange
from fem.femspace import FEMSpace
from fem.assembler import Assembler
from fem.linearsolver import *
import utils.logger as log 

logger = log.setup_logger(__name__, level = 'info')

class HeatProblem:
    def __init__(self, femspace: FEMSpace, func, t0: float, T: float, dirichlet_bc: dict, icond: np.ndarray):
        """
        FEM solver for the time-dependent heat equation:
            - 1D: u_t(x, t) - u_xx(x, t)) = f(x, t),   x ∈ Ω, t ∈ (t0, T)
            - 2D: u_t(x, y, t) - ∇·(∇u(x, y, t)) = f(x, y, t),   (x, y) ∈ Ω, t ∈ (t0, T)
        with:
            - Dirichlet boundary conditions specified at all times
            - Initial condition 
                - 1D: u(x, t0) = u0(x)
                - 2D: u(x, y, t0) = u0(x, y)
        The semi-discrete system (after spatial FEM discretization) is:
            M u'(t) + A u(t) = F(t)
        where:
            - M : mass matrix
            - A : stiffness matrix
            - F(t) : load vector due to source term f(x, t) or f(x, y, t)

        Parameters
        ----------
        femspace : FEMSpace
            FEM space object
        func : callable
            Source term f(x, t) for 1D or f(x, y, t) for 2D
        t0 : float
            Initial time
        T : float
            Final time
        dirichlet_bc : dict
            Dirichlet BCs at all time steps. Keys are global node indices;
            values are np.ndarray of length n_steps: {0: [0.0, 0.45, 3.4], ...}
        icond : np.ndarray
            Initial condition vector at t0
        """
        self.femspace = femspace
        self.f = func
        self.dirichlet = dirichlet_bc
        self.initial = icond
        self.t0 = t0
        self.T = T
        self.dim = femspace.dim
        self.assembler = Assembler(self.femspace)
        self.mass_matrix, self.stiffness_matrix = self._assemble_space()
        if self.dim == 2:
            self.F = lambda t: self.assembler.global_load_vector(lambda x, y: self.f(x, y, t))
        else:
            self.F = lambda t: self.assembler.global_load_vector(lambda x: self.f(x, t))

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

    def lift(self, lift: str = 'nodal', solver: LinearSolver = DirectSolver(), **kwargs) -> np.ndarray:
        return None

    def _theta(self, dt: float, lift: str = 'nodal', theta: float = 0.5, solver: LinearSolver = DirectSolver(), **kwargs) -> np.ndarray:

        # Total number of time steps
        nsteps = int((self.T - self.t0)/dt)

        # Pre-allocate solution array: columns are time steps
        solution = np.zeros((self.mass_matrix.shape[0], nsteps + 1))

        # Initial solution
        solution[:, 0] = self.initial

        logger.debug(f"Using θ-method time-stepping | Starting time integration | nsteps = {nsteps}, dt = {dt}")

        # Time-stepping loop with tqdm
        step = 0
        t = self.t0
        u_initial = self.initial
        load_initial = self.F(t)
        with trange(nsteps, desc = "\033[92mHeat Solver\033[0m", unit="step",ascii = "░▒█", ncols = 100, disable = not sys.stdout.isatty()) as pbar:
            for step in pbar:
                pbar.set_postfix_str(f"\033[93mt={t:.3e}\033[0m")
                step += 1
                t += dt
                load_vector = self.F(t)
                lhs = self.mass_matrix + theta*dt*self.stiffness_matrix
                rhs = (self.mass_matrix - (1 - theta)*dt*self.stiffness_matrix)@u_initial + dt*(theta*load_vector + (1 - theta)*load_initial).ravel()
                dirichlet = {k: v[step] for k, v in self.dirichlet.items()}
                lhs, rhs = self.assembler.apply_Dirichlet_bc(lhs, rhs, dirichlet)
                u = solver.solve(lhs, rhs, **kwargs)
                solution[:, step + 1] = u
                u_initial = u
                load_initial = load_vector
        return solution

    def solve(self, dt: float, lift: str = 'nodal', theta: float = 0.5, solver: LinearSolver = DirectSolver(), **kwargs) -> np.ndarray:

        if lift == 'nodal':
            solution = self._theta(dt, lift = lift, theta = theta, solver = solver, **kwargs)
        elif lift == 'harmonic':
            boundary = bd.HarmonicDirichletBC(self.femspace, self.dirichlet)
            lift_vector = boundary.lift
            solution[:, 0] += lift_vector
        elif lift == 'parabolic':
            boundary = 
            lift_vector = boundary.lift
            solution[:, 0] += lift_vector[:, 0]
        else:
            raise ValueError(f"Unsupported lift method: {lift}")
        
        return None