import sys
import numpy as np
from tqdm import trange
from fom.heat import HeatProblem
import timestepper as ts
import logger as log 

logger = log.setup_logger(__name__, level = 'info')

class HeatROM:
    def __init__(self, fom: HeatProblem, basis: np.ndarray, tstepper: str = 'Theta', theta: float = 0.5):
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
        fom : HeatProblem
            Full-order FEM solver instance
        basis : np.ndarray
            Precomputed reduced basis (self.M.shape[0] x r), columns form orthonormal basis.
        tstepper : str
            Time integration method (only 'Theta')
        theta : float, optional
            Parameter for the θ-method, 0 < θ ≤ 1. Only used if tstepper='Theta'.
            Common values:
                - θ = 1.0  → Backward Euler
                - θ = 0.5  → Crank-Nicolson
            Default is 0.5.
        """
        self.fom = fom
        self.basis = basis
        self.r = self.basis[1]
        self.tstepper = tstepper
        self.theta = theta
        self.assembler, self.M, self.A = fom.assemble_space()
        self.Mr = basis.T@self.M@basis
        self.Ar = basis.T@self.A@basis
        self.initial = basis.T@self.fom.initial
    
    def construct_bc(self):

        'This function reconstruct the reduced solution obtained from neigh subdomains to get interface '
        'boundary data and then project again to use in ROWR'

        self.fom.dirichlet



        return None
    
    def construct_initial(self) -> np.ndarray:
        """
        Project the full-order initial condition onto the reduced space.

        This method constructs the initial condition for the reduced-order model
        by projecting the full-order FEM initial condition vector onto the 
        precomputed reduced basis. The resulting vector is ready to be used 
        in time integration of the reduced system.

        Returns
        -------
        np.ndarray
            Reduced initial condition vector of shape (r,), where r is the
            number of reduced basis vectors.
        """
        return self.basis.T @ self.fom.initial


    def solve(self) -> np.ndarray:
        """
        Solve the PDE using the specified time-stepping method.

        Returns
        -------
        solution : np.ndarray
            Each column corresponds to the solution at a specific time step:
            solution[:, i] = u(x, t_i), with the first column representing the initial condition at t0.
        """

        nsteps = self.fom.ntime - 1

        # Pre-allocate solution array: columns are time steps
        solution = np.zeros((self.r, self.fom.ntime))

        logger.debug(f"Using {self.tstepper} time-stepping | Starting time integration | nsteps = {nsteps}, dt = {self.fom.dt}")

        # Initial solution
        u_n = self.initial.copy()
        solution[:, 0] = u_n

        # Initialize previous step matrices and load vector for Crank-Nicolson
        if self.fom.dim == 2:
            func = lambda x, y: self.fom.f(x, y, self.fom.t0)
        else:  # 1D
            func = lambda x: self.fom.f(x, self.fom.t0)

        F_prev = self.basis.T@self.assembler.global_load_vector(func)@self.basis

        A_prev, M_prev = self.Ar, self.Mr

        # Select time-stepper
        if self.tstepper == 'Theta':
            stepper = ts.Theta(M = self.Mr, A = self.Ar, assembler = self.assembler, f = self.f, dt = self.fom.dt, t0 = self.fom.t0, dirichlet_bc = self.dirichlet, theta = self.theta)
        else:
            raise ValueError(f"Unknown time-stepper: {self.tstepper}")

        # Time-stepping loop with tqdm
        t_n = self.fom.t0
        with trange(nsteps, desc = "\033[92mHeat Solver\033[0m", unit="step",ascii = "░▒█", ncols = 100, disable = not sys.stdout.isatty()) as pbar:
            for step in pbar:
                pbar.set_postfix_str(f"\033[93mt={t_n:.3e}\033[0m")  # yellow t
                if self.tstepper == 'Theta':
                    u_n, A_prev, M_prev, F_prev = stepper.step(u_n, F_prev, t_n, A_prev, M_prev)
                t_n += self.fom.dt
                solution[:, step + 1] = u_n.copy()
        return solution


