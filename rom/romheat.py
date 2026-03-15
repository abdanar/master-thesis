import sys
import numpy as np
from tqdm import trange
from fom.heat import HeatProblem
import fem.timestepper as ts
import utils.logger as log 
from fem.femspace import FEMSpace
from fem.assembler import Assembler
from fem.linearsolver import LinearSolver, DirectSolver, JacobiSolver, CGSolver
from rom.reducedstepper import ModifiedTheta

logger = log.setup_logger(__name__, level = 'info')

class TestHeatROM:
    def __init__(self, femspace: FEMSpace, basis: np.ndarray, func, dt: float, t0: float, T: float, dirichlet_bc: dict, icond: np.ndarray, tstepper: str = 'Theta', theta: float = 0.5):
        """
        Reduced-order model (ROM) solver for the time-dependent heat equation.

        The ROM uses a precomputed reduced basis to project the full-order FEM system
        onto a smaller system and solve efficiently in reduced coordinates. 

        The reduced system is:
            M_r u_r'(t) + A_r u_r(t) = F_r(t)

        where:
            - M_r : reduced mass matrix
            - A_r : reduced stiffness matrix
            - F_r(t) : reduced source term
            - u_r(t) : reduced solution vector

        Parameters
        ----------
        femspace : FEMSpace
            FEM space object
        basis : np.ndarray
            Precomputed reduced basis (self.M.shape[0] x r), columns form orthonormal basis.
        func : callable
            Source term f(x, t) for 1D or f(x, y, t) for 2D
        dt : float
            Time step size
        t0 : float
            Initial time
        T : float
            Final time
        dirichlet_bc : dict
            Projected Dirichlet BCs at all time steps. Keys are global node indices;
            values are np.ndarray of length n_steps: {0: [0.0, 0.45, 3.4], ...}
        icond : np.ndarray
            Projected initial condition of FOM vector at t0
        tstepper : str
            Time integration method (only 'Theta')
        theta : float, optional
            Parameter for the θ-method, 0 < θ ≤ 1. Only used if tstepper='Theta'.
            Common values:
                - θ = 1.0  → Backward Euler
                - θ = 0.5  → Crank-Nicolson
            Default is 0.5.
        """
        self.femspace = femspace
        self.f = func
        self.dt = dt
        self.t0 = t0
        self.T = T
        self.dirichlet = dirichlet_bc
        self.initial = icond
        self.tstepper = tstepper
        self.theta = theta
        self.ntime = int((T - t0)/dt) + 1 # total number of time nodes (including boundaries)
        self.dim = femspace.dim
        self.basis = basis
        self.r = basis.shape[1] # reduced order
        self.assembler, self.M, self.A = self.assemble_space()
        self.Mr = basis.T@self.M@basis
        self.Ar = basis.T@self.A@basis
        self.fom_lhs = self.M + self.theta*self.dt*self.A
        if self.dim == 2:
            self.F = lambda t: Assembler(self.femspace).global_load_vector(lambda x, y: self.f(x, y, t))
            self.Fr = lambda t: basis.T@self.assembler.global_load_vector(lambda x, y: self.f(x, y, t))
        else:
            self.F = lambda t: Assembler(self.femspace).global_load_vector(lambda x: self.f(x, t))
            self.Fr = lambda t: basis.T@self.assembler.global_load_vector(lambda x: self.f(x, t))

    def assemble_space(self):
        """
        Assemble space-dependent matrices (M, A) using the assembler.
        Returns:
            M : np.ndarray
                Mass matrix
            A : np.ndarray
                Stiffness matrix
            assembler : Assembler
                FEM assembler object
        """
        assembler = Assembler(self.femspace)

        if self.dim == 2:
            diffusion = lambda x, y: np.eye(2)
            reaction  = lambda x, y: 1
        else:  # 1D
            diffusion = lambda x: 1
            reaction  = lambda x: 1

        logger.debug("Assembling spatial FEM matrices")

        M = assembler.global_mass_matrix(reaction = reaction)
        logger.debug(f"Mass matrix assembled with shape {M.shape}")

        A = assembler.global_stiffness_matrix(diffusion = diffusion)
        logger.debug(f"Stiffness matrix assembled with shape {A.shape}")

        return assembler, M, A

    def reconstruct(self, u_r: np.ndarray) -> np.ndarray:
        """
        Reconstruct the full-order solution from the reduced-order solution.

        Parameters
        ----------
        u_r : np.ndarray
            Reduced solution vector of shape (r, nteps), where r is the reduced order.

        Returns
        -------
        u_full : np.ndarray
            Reconstructed full-order solution vector of shape (n, nteps), where n is the number of DOFs in the full model.
        """
        return self.basis @ u_r
    
    def solve(self) -> np.ndarray:
        """
        Solve the PDE using the specified time-stepping method.

        Returns
        -------
        solution : np.ndarray
            Each column corresponds to the solution at a specific time step:
            solution[:, i] = u(x, t_i), with the first column representing the initial condition at t0.
        """
        nsteps = self.ntime - 1

        # Pre-allocate solution array: columns are time steps
        solution = np.zeros((self.M.shape[0], self.ntime))
        logger.debug(f"Using {self.tstepper} time-stepping | Starting time integration | nsteps = {nsteps}, dt = {self.dt}")

        # Initial solution
        u_n = self.initial.copy()
        solution[:, 0] = u_n

        F_prev = self.F(self.t0)
        A_prev, M_prev = self.A, self.M

        # Select time-stepper
        if self.tstepper == 'Theta':
            stepper = ModifiedTheta(M = self.M, A = self.A, assembler = self.assembler, F = self.F, basis = self.basis, dt = self.dt, t0 = self.t0, dirichlet_bc = self.dirichlet, theta = self.theta)
        else:
            raise ValueError(f"Unknown time-stepper: {self.tstepper}")

        # Time-stepping loop with tqdm
        t_n = self.t0
        with trange(nsteps, desc = "\033[92mHeat Solver\033[0m", unit="step",ascii = "░▒█", ncols = 100, disable = not sys.stdout.isatty()) as pbar:
            for step in pbar:
                pbar.set_postfix_str(f"\033[93mt={t_n:.3e}\033[0m")  # yellow t
                if self.tstepper == 'Theta':
                    u_n, A_prev, M_prev, F_prev = stepper.step(u_n, F_prev, t_n, A_prev, M_prev)
                t_n += self.dt
                solution[:, step + 1] = u_n.copy()
        return solution