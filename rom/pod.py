import numpy as np
import utils.logger as log 
import scipy.sparse as sp
from fom.heat import HeatProblem
from fem.linearsolver import DirectSolver, LinearSolver
from typing import Optional
from scipy.linalg import cholesky, solve_triangular

logger = log.setup_logger(__name__, level = 'info')

class POD:
    def __init__(self, heat_problem: HeatProblem, ntime: int, lift: str, theta: float, r: int, weight: Optional[np.ndarray] = None, energy_tol: float = 0.999, solver: LinearSolver = DirectSolver()):
        """
        Proper Orthogonal Decomposition (POD) for model order reduction of the Heat problem.

        Parameters
        ----------
        heat_problem : HeatProblem
            The full-order Heat problem to be reduced.
        ntime : int
            Number of time steps (snapshots).
        lift : str
            Lifting method.
        theta : float
            Theta parameter for the time integration scheme.
        r : int
            Fixed reduced dimension.
        weight : np.ndarray, optional
            Weight matrix for the inner product (used in weighted POD).
        energy_tol : float
            Energy threshold (used if r is None).
        solver : LinearSolver
            Linear solver to use for the full-order problem.
        """
        self.hp = heat_problem
        self.ntime = ntime
        self.lift = lift
        self.theta = theta
        self.solver = solver
        self.r = r
        self.energy_tol = energy_tol
        self.weight = weight

        # Compute POD basis
        self.V, self.singular_values = self.compute_modes()

    def compute_snapshots(self):
        return self.hp.solve(time_grid=np.linspace(self.hp.t0, self.hp.T, self.ntime), lift=self.lift, theta=self.theta, solver=self.solver, homogeneous=True) # shape (n_interior, n_time)

    def compute_modes(self):
        logger.info("Computing snapshots for POD...")
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
