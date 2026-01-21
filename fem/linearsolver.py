from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from scipy.sparse import sparray
import scipy as sc

# not correct change entirely

MatrixValue = np.ndarray | sparray

class LinearSolver(ABC):
    
    def setup(self, A: MatrixValue) -> None:
        """
        Optional preprocessing step (e.g. factorization).
        Default: do nothing.
        """
        pass

    @abstractmethod
    def solve(self, A: MatrixValue, b: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Solve the linear system A x = b.

        Parameters
        ----------
        A : MatrixValue
            System matrix (dense or sparse).
        b : np.ndarray
            Right-hand side vector.
        **kwargs :
            Solver-specific options (e.g., tolerance, preconditioner).

        Returns
        -------
        x : np.ndarray
            Solution vector.
        """
        pass

class DirectSolver(LinearSolver):
    """
    Direct linear solver supporting both dense and sparse matrices with LU caching.

    - Dense matrices (`np.ndarray`) use `scipy.linalg.lu_factor` and `lu_solve`.
    - Sparse matrices: uses scipy.sparse.linalg.splu
      LU factorization is cached for repeated solves.

    Example
    -------
    >>> solver = DirectSolver()
    >>> x = solver.solve(A, b)  # A can be dense or sparse
    """
    def __init__(self):
        # Dense LU factorization cache
        self._lu_dense: Any = None
        self._A_dense_id: int | None = None

        # Sparse LU factorization cache
        self._lu_sparse: Any = None
        self._A_sparse_id: int | None = None

    def setup(self, A: MatrixValue) -> None:
        """
        Precompute LU factorization for dense or sparse matrices.

        Parameters
        ----------
        A : MatrixValue
            Dense or sparse matrix to factorize.
        """
        if sc.sparse.issparse(A):
            self._lu_sparse = sc.sparse.linalg.splu(A)
            self._A_sparse_id = id(A)
        else:
            self._lu_dense = sc.linalg.lu_factor(A)
            self._A_dense_id = id(A)

    def solve(self, A: MatrixValue, b: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Solve the linear system Ax = b using direct LU methods.

        Parameters
        ----------
        A : MatrixValue
            Dense or sparse system matrix.
        b : np.ndarray
            Right-hand side vector.
        **kwargs :
            Solver-specific options (ignored here, placeholder for subclasses).

        Returns
        -------
        x : np.ndarray
            Solution vector.
        """
        if sc.sparse.issparse(A):
            if self._lu_sparse is None or self._A_sparse_id != id(A):
                self.setup(A)
            return self._lu_sparse.solve(b)
        else:
            if self._lu_dense is None or self._A_dense_id != id(A):
                self.setup(A)
            return sc.linalg.lu_solve(self._lu_dense, b)