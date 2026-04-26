import numpy as np
from utils.logger import get_logger
from fom.heat_fom import HeatProblem
from fem.linearsolver import DirectSolver, LinearSolver
from typing import Optional, Literal
from scipy.linalg import cholesky, solve_triangular
logger = get_logger(__name__)

class POD:
    def __init__(self, heat_problem: HeatProblem, time_grid: np.ndarray, lift: Literal['nodal', 'harmonic', 'parabolic'], theta: float, r: int, 
                g: Optional[np.ndarray] = None, weight: Optional[np.ndarray] = None, energy_tol: float = 0.999, solver: LinearSolver = DirectSolver()):
        """
        Proper Orthogonal Decomposition (POD) for model order reduction of the Heat problem.

        Parameters
        ----------
        heat_problem : HeatProblem
            The full-order Heat problem to be reduced.
        time_grid : np.ndarray
            Array of time points for the simulation.
        lift : Literal['nodal', 'harmonic', 'parabolic']
            Lifting method.
        theta : float
            Theta parameter for the time integration scheme.
        r : int
            Fixed reduced dimension.
        g : np.ndarray, optional
            Numpy array of shape (nbdnodes, len(time_grid)) containing the boundary values at each time step.
        weight : np.ndarray, optional
            Weight matrix for the inner product (used in weighted POD).
        energy_tol : float
            Energy threshold (used if r is None).
        solver : LinearSolver
            Linear solver to use for the full-order problem.
        """
        self.hp = heat_problem
        self.time_grid = time_grid
        self.lift = lift
        self.theta = theta
        self.solver = solver
        self.r = r
        self.energy_tol = energy_tol
        self.weight = weight
        self.g = g
        self.ntime = len(time_grid)

    def compute_snapshots(self):
        if isinstance(self.hp.g, np.ndarray):
            assert self.g is not None, "Boundary condition array g must be provided if hp.g is a numpy array."
            return self.hp.solve(time_grid=self.time_grid, lift=self.lift, theta=self.theta, solver=self.solver, g_new = self.g, homogeneous=True) # shape (n_interior, n_time)
        else:
            return self.hp.solve(time_grid=self.time_grid, lift=self.lift, theta=self.theta, solver=self.solver, homogeneous=True) # shape (n_interior, n_time)
    
    def compute_modes(self):
        logger.debug("Computing snapshots for POD...")
        snapshot_matrix = self.compute_snapshots()

        if self.weight is None:
            V, S, _ = np.linalg.svd(snapshot_matrix, full_matrices=False)
        else:
            L = cholesky(self.weight, lower=True)
            V, S, _ = np.linalg.svd(L @ snapshot_matrix, full_matrices=False)

        if self.r is not None:
            r = self.r
        else:
            energy = np.cumsum(S**2) / np.sum(S**2)
            r = np.searchsorted(energy, self.energy_tol) + 1

        if self.weight is not None:
            return solve_triangular(L, V[:, :r], lower=True), S[:r]
        else:
            return V[:, :r], S[:r]
