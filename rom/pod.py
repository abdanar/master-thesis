import numpy as np
from typing import Optional
from scipy.linalg import cholesky, solve_triangular
from utils.logger import get_logger
logger = get_logger(__name__)

class POD:
    def __init__(self, snapshots: np.ndarray, r: Optional[int] = None, energy_tol: float = 0.999, weight: Optional[np.ndarray] = None):
        """
        This class provides general functionality for Proper Orthogonal Decomposition (POD).

        Parameters
        ----------
        snapshots : np.ndarray
            Array of snapshots for the POD. Each column of the array should correspond 
            to a snapshot of the system at a given parameter.
        r : int
            Fixed reduced dimension. If None, the reduced dimension will be determined 
            based on the energy content of the singular values.
        energy_tol : float
            Energy threshold for determining the reduced dimension when r is None. The 
            reduced dimension will be chosen such that the cumulative energy of the 
            retained modes is at least `energy_tol`.
        weight : np.ndarray, optional
            Weight matrix for the inner product (used in weighted POD). If None, the 
            standard Euclidean inner product will be used.
        r : int
            Fixed reduced dimension.
        """
        assert snapshots.ndim == 2, "Snapshots should be a 2D array where each column is a snapshot."
        assert r is None or (isinstance(r, int) and r > 0), "r should be a positive integer or None."
        assert 0 < energy_tol <= 1, "energy_tol should be in the range (0, 1]."
        #assert weight is None or (isinstance(weight, np.ndarray) and weight.ndim == 2 and weight.shape[0] == weight.shape[1] == snapshots.shape[0]), "Weight should be a square matrix of the same size as the number of rows in snapshots."
        self.weight = weight
        self.snapshots = snapshots
        self.energy_tol = energy_tol
        self.r = r if r is not None else self.rank()

    def _compute_svd(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the Singular Value Decomposition (SVD) of the weighted snapshots.

        This method computes the SVD of the weighted snapshot matrix, which is essential for
        determining the POD modes and their corresponding singular values. The SVD is computed
        using the weighted snapshots, which are obtained by applying the Cholesky factor of the
        weight matrix to the original snapshots. If the weight is None, it defaults to the
        standard SVD of the snapshots.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            U : np.ndarray
                Left singular vectors (POD modes).
            S : np.ndarray
                Singular values.
            Vh : np.ndarray
                Right singular vectors (transposed).
        """
        if self.weight is not None:
            L = cholesky(self.weight, lower = True) # gives you the lower triangular matrix L such that L @ L.T = weight
            weighted_snapshots = L.T @ self.snapshots
        else:
            weighted_snapshots = self.snapshots
        return np.linalg.svd(weighted_snapshots, full_matrices = False)

    def rank(self) -> int:
        """
        Determine the reduced dimension r based on the energy content of the singular values.

        Returns
        -------
        int
            Reduced dimension.
        """
        assert self.energy_tol is not None, "Energy tolerance must be specified to determine the rank when r is None."
        _, S, _ = self._compute_svd()
        energy = np.cumsum(S**2) / np.sum(S**2)
        return int(np.searchsorted(energy, self.energy_tol) + 1)
         
    def snapshot_differences(self) -> np.ndarray:
        """
        Compute the difference between consecutive snapshots. 
        
        This can be useful for methods that require time derivatives or 
        differences, such as Difference Quotients(DQ) POD. It computes the 
        difference between each snapshot and the previous one, resulting 
        in an array of shape (n, nsnaps - 1), where n is the number of 
        rows in the snapshots and nsnaps is the number of snapshots.

        Returns
        -------
        np.ndarray
            Array of differences between consecutive snapshots.
        """   
        return self.snapshots[:, 1:] - self.snapshots[:, :-1]     
    
    def dq_snapshots(self, dt: float = 1.0) -> np.ndarray:
        """
        Construct the snapshots of the POD based on the Difference Quotients (DQ) 
        of the snapshots with a given uniform step size. 

        This function concatenates the original snapshots with the difference quotients 
        of the snapshots, which are computed by dividing the differences between consecutive 
        snapshots by the time step size `dt`.

        Returns
        -------
        np.ndarray
            Array of snapshots including the original snapshots and their difference quotients.
        """   
        return np.column_stack((self.snapshots, self.snapshot_differences()/dt))
    
    def correlation_matrix(self) -> np.ndarray:
        """
        Compute the correlation matrix of the snapshots. 
        
        It computes the following inner product
            K_ij = <snapshot_i, snapshot_j>_{weight},
        where <.,.>_{weight} is the weighted inner product defined 
        by the weight matrix. If the `weight` is None, it defaults 
        to the standard Euclidean inner product.

        Returns
        -------
        np.ndarray
            Correlation matrix of the snapshots.
        """
        # Get the number of snapshots
        nsnaps = self.snapshots.shape[1]
        # Compute the correlation matrix using the weighted inner product
        if self.weight is not None:
            return self.snapshots.T @ self.weight @ self.snapshots
        else:
            return self.snapshots.T @ self.snapshots

    def truncation_error(self, relative: bool = True) -> float:
        """
        Compute the truncation error based on the singular values of the 
        snapshot matrix. 
        
        The truncation error is defined as the ratio of the sum of the 
        squares of the discarded singular values to the total energy 
        (sum of squares of all singular values). If `relative` is False, 
        it returns the absolute truncation error, which is simply the 
        sum of the squares of the discarded singular values.

        Parameters
        ----------
        relative : bool
            If True, return the relative truncation error. If False, 
            return the absolute truncation error.

        Returns
        -------
        float
            Truncation error.
        """
        _, S, _ = self._compute_svd()
        sumsq = np.sum(S[self.r:]**2)
        return sumsq/np.sum(S**2) if relative else sumsq

    def basis(self) -> np.ndarray:
        """
        Compute the POD basis of rank r. 
        
        This method computes the POD modes based on the snapshots and the 
        specified parameters (r, energy_tol, weight). It returns the first r 
        modes of the POD basis, where r is determined either by the fixed 
        dimension or by the energy content of the singular values.

        Returns
        -------
        np.ndarray
            POD basis of rank r.
        """
        U, _, _ = self._compute_svd()
        if self.weight is None:
            return U[:, :self.r]
        else:
            L = cholesky(self.weight, lower = True)
            return solve_triangular(L.T, U[:, :self.r], lower = False)