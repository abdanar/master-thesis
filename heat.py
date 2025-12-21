import sys
import numpy as np
from tqdm import trange
from mesh import Mesh
from assembler import Assembler
import timestepper as ts
import logger as log 

logger = log.setup_logger(__name__, level = 'info')

class HeatProblem:
    def __init__(self, mesh: Mesh, func, dt: float, t0: float, T: float, dirichlet_bc: dict, icond: np.ndarray, tstepper: str = 'BackwardEuler', theta: float = 0.5):

        """
        FEM solver for the time-dependent heat equation:
            u_t(x, y, t) - ∇·(∇u(x, y, t)) = f(x, y, t),   (x, y) ∈ Ω, t ∈ (t0, T)
        with:
            - Dirichlet boundary conditions specified at all times
            - Initial condition u(x, y, t0) = u0(x, y)
        The semi-discrete system (after spatial FEM discretization) is:
            M u'(t) + A u(t) = F(t)
        where:
            - M : mass matrix
            - A : stiffness matrix
            - F(t) : load vector due to source term f(x, y, t)

        Parameters
        ----------
        mesh : Mesh
            FEM mesh object
        func : callable
            Source term f(x, y, t)
        dt : float
            Time step size
        t0 : float
            Initial time
        T : float
            Final time
        dirichlet_bc : dict
            Dirichlet BCs at all time steps. Keys are global node indices;
            values are np.ndarray of length n_steps: {0: [0.0, 0.45, 3.4], ...}
        icond : np.ndarray
            Initial condition vector at t0
        tstepper : str
            Time integration method ('BackwardEuler', 'CrankNicolson', or 'Theta')
        theta : float, optional
            Parameter for the θ-method, 0 < θ ≤ 1. Only used if tstepper='Theta'.
            Common values:
                - θ = 1.0  → Backward Euler
                - θ = 0.5  → Crank-Nicolson
            Default is 0.5.
        """

        self.mesh = mesh
        self.f = func
        self.dt = dt
        self.t0 = t0
        self.T = T
        self.dirichlet = dirichlet_bc
        self.initial = icond
        self.tstepper = tstepper
        self.theta = theta
        self.ntime = int((T - t0)/dt) + 1 # total number of time nodes (including boundaries)

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

        assembler = Assembler(self.mesh)
        diffusion = lambda x, y: np.eye(2)
        reaction = lambda x, y: 1

        logger.debug("Assembling spatial FEM matrices")

        M = assembler.global_mass_matrix(reaction = reaction)
        logger.debug(f"Mass matrix assembled with shape {M.shape}")

        A = assembler.global_stiffness_matrix(diffusion = diffusion)
        logger.debug(f"Stiffness matrix assembled with shape {A.shape}")

        return assembler, M, A

    def solve(self) -> np.ndarray:

        """
        Solve the PDE using the specified time-stepping method.

        Returns
        -------
        solution : np.ndarray
            Each column corresponds to the solution at a specific time step:
            solution[:, i] = u(x, t_i), with the first column representing the initial condition at t0.
        """

        # Assemble FEM matrices
        assembler, M, A = self.assemble_space()
        nsteps = self.ntime - 1

        # Pre-allocate solution array: columns are time steps
        solution = np.zeros((M.shape[0], self.ntime))

        logger.debug(f"Using {self.tstepper} time-stepping | Starting time integration | nsteps = {nsteps}, dt = {self.dt}")

        # Initial solution
        u_n = self.initial.copy()
        solution[:, 0] = u_n

        # Initialize previous step matrices and load vector for Crank-Nicolson
        F_prev = assembler.global_load_vector(lambda x, y: self.f(x, y, self.t0))
        A_prev, M_prev = A, M

        # Select time-stepper
        if self.tstepper == 'BackwardEuler':
            stepper = ts.BackwardEuler(M = lambda t: M, A = lambda t: A, assembler = assembler, f = self.f,
                                    dt = self.dt, t0 = self.t0, dirichlet_bc = self.dirichlet)
        elif self.tstepper == 'CrankNicolson':
            stepper = ts.CrankNicolson(M =  lambda t: M, A = lambda t: A, assembler = assembler, f = self.f,
                                    dt = self.dt, t0 = self.t0, dirichlet_bc = self.dirichlet)
        elif self.tstepper == 'Theta':
            stepper = ts.Theta(M =  lambda t: M, A = lambda t: A, assembler = assembler, f = self.f,
                                    dt = self.dt, t0 = self.t0, dirichlet_bc = self.dirichlet, theta = self.theta)
        else:
            raise ValueError(f"Unknown time-stepper: {self.tstepper}")

        # Time-stepping loop with tqdm
        t_n = self.t0
        with trange(nsteps, desc = "\033[92mHeat Solver\033[0m", unit="step",ascii = "░▒█", ncols = 100, disable = not sys.stdout.isatty()) as pbar:
            for step in pbar:
                pbar.set_postfix_str(f"\033[93mt={t_n:.3e}\033[0m")  # yellow t
                if self.tstepper == 'BackwardEuler':
                    u_n = stepper.step(u_n, t_n)
                elif self.tstepper == 'CrankNicolson':
                    u_n, A_prev, M_prev, F_prev = stepper.step(u_n, A_prev, M_prev, F_prev, t_n)
                elif self.tstepper == 'Theta':
                    u_n, A_prev, M_prev, F_prev = stepper.step(u_n, A_prev, M_prev, F_prev, t_n)
                t_n += self.dt
                solution[:, step + 1] = u_n.copy()
        return solution