import numpy as np
from fom.heat import HeatProblem
from fem.linearsolver import DirectSolver, LinearSolver

class POD:
    def __init__(
        self,
        heat_problem: HeatProblem,
        ntime: int,
        lift: str,
        r: int,
        energy_tol: float = 0.999,
        solver: LinearSolver = DirectSolver(),
    ):
        """
        Mass-weighted POD for FEM heat problem.

        Parameters
        ----------
        heat_problem : HeatProblem
        ntime : int
            Number of time steps (snapshots).
        lift : str
            Lifting method.
        r : int
            Fixed reduced dimension.
        energy_tol : float
            Energy threshold (used if r is None).
        solver : LinearSolver
        """
        self.hp = heat_problem
        self.ntime = ntime
        self.lift = lift
        self.solver = solver

        self.r = r
        self.energy_tol = energy_tol

        # Compute POD basis
        self.V, self.singular_values = self.compute_modes()

    def compute_snapshots(self):
        # Boundary nodes
        if isinstance(self.hp.g, dict):
            boundary_nodes = np.array(list(self.hp.g.keys()), dtype=np.int64)
        else:
            boundary_nodes = np.fromiter(
                self.hp.femspace.mesh.boundary_nodes(), dtype=np.int64
            )

        # Interior nodes
        mask = np.ones(self.hp.femspace.nnodes, dtype=bool)
        mask[boundary_nodes] = False
        interior_nodes = np.nonzero(mask)[0]

        # Solve full problem
        solution = self.hp.solve(
            ntime=self.ntime,
            lift=self.lift,
            solver=self.solver,
        )

        snapshots = solution[interior_nodes, :]

        return snapshots, interior_nodes

    def compute_modes(self):
        snapshots, interior_nodes = self.compute_snapshots()

        # Interior mass matrix (CSR)
        M = self.hp.mass_matrix.tocsr()[np.ix_(interior_nodes, interior_nodes)]

        # Correlation matrix (mass-weighted)
        C = snapshots.T @ M @ snapshots  # shape (ntime x ntime)

        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(C)

        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Truncate to the first r modes
        eigvals_r = eigvals[:self.r]
        eigvecs_r = eigvecs[:, :self.r]

        # Build modes
        V = snapshots @ eigvecs_r  # shape (N_int x r)

        # M-orthonormalization (stable)
        G = V.T @ M @ V
        L = np.linalg.cholesky(G)
        V = V @ np.linalg.inv(L)

        return V, np.sqrt(eigvals_r)