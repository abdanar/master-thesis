from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from scipy.sparse import sparray
import scipy as sc
from utils.logger import setup_logger
from scipy.sparse.linalg import cg, LinearOperator

logger = setup_logger(__name__, level = 'info')

#----------------- Linear Solvers for Finite Element Method (FEM) -------------------------
# The `LinearSolver` class is an abstract base class that defines the interface for linear solvers
# used in finite element methods. It provides a common structure for both direct and iterative
# solvers, allowing for flexible implementation of various algorithms. The available solvers include:
#
# - `DirectSolver`: A direct solver that computes the exact solution using LU factorization,
#   supporting both dense and sparse matrices.
#
# - `IterativeSolver`: A base class for iterative solvers that provides common functionality and
#   an iteration error criterion. These solvers do not perform direct factorization and rely on 
#   convergence criteria to determine when to stop iterating. Available iterative solvers include:
#
#   - `JacobiSolver`: An implementation of the Jacobi iterative method, which can be used with
#     both dense and sparse matrices. It supports weighted (relaxed) iteration for improved
#     convergence.
#
#   - `CGSolver`: An implementation of the Conjugate Gradient method, suitable for
#     symmetric positive definite matrices, and can also handle both dense and sparse formats.
#
# The class includes methods for setting up the solver with a given matrix, resetting any cached 
# state, and solving linear systems. Subclasses must implement the `solve` method, which takes a 
# system matrix and right-hand side vector and returns the solution. The design allows for easy 
# integration of new solvers and supports both dense and sparse matrices, making it suitable for 
# a wide range of FEM applications.
# --------------------------------------------------------------------------------------

class LinearSolver(ABC):
    """
    Abstract base class for linear solvers.

    This class defines the common interface for all linear solvers, whether direct or iterative.
    Subclasses must implement the `solve` method. Optionally, they can override `setup` to 
    perform preprocessing such as factorization or caching.

    Attributes
    ----------
    _A_id : int
        Internal ID of the last matrix passed to `setup`, used to track changes.
    _is_sparse : bool
        Flag indicating whether the system matrix is sparse (`True`) or dense (`False`).

    Notes
    -----
    1. Iterative solvers may override `setup` to store matrix properties such as sparsity or
       precompute auxiliary data.
    2. Direct solvers may override `setup` to compute and cache matrix factorizations.
    3. `reset` can be overridden by subclasses to clear cached state if needed.
    4. Subclasses should check if the matrix `A` has changed by comparing `id(A)` with `_A_id`.

    Examples
    --------
    >>> from numpy import array
    >>> class DummySolver(LinearSolver):
    ...     def solve(self, A, b, **kwargs):
    ...         return b.copy()  # trivial solver
    >>> solver = DummySolver()
    >>> A = array([[2, 1], [1, 3]])
    >>> b = array([1, 2])
    >>> x = solver.solve(A, b)
    >>> print(x)
    [1 2]
    """
    def setup(self, A: np.ndarray | sparray) -> None:
        """
        Optional preprocessing step for the solver.

        Stores basic matrix information such as ID and sparsity. 
        Subclasses can override to perform actual preprocessing (e.g., LU factorization).

        Parameters
        ----------
        A : np.ndarray or scipy.sparse.sparray
            Dense (`np.ndarray`) or sparse (`sparray`) system matrix.

        Notes
        -----
        - By default, only `_A_id` and `_is_sparse` are set.
        - Subclasses may call `super().setup(A)` to preserve these properties.
        """
        self._A_id = id(A)
        self._is_sparse = sc.sparse.issparse(A)

    def reset(self):
        """
        Invalidate any cached solver state.

        Notes
        -----
        - Base class does nothing.
        - Subclasses can override this method to clear factorization caches, iteration counters, etc.
        """
        pass

    @abstractmethod
    def solve(self, A: np.ndarray | sparray, b: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Solve the linear system Ax = b.

        Parameters
        ----------
        A : np.ndarray or scipy.sparse.sparray
            Dense (`np.ndarray`) or sparse (`sparray`) system matrix.
        b : np.ndarray
            Right-hand side vector.
        **kwargs : Any
            Solver-specific options (e.g., tolerance, maximum iterations, preconditioner).

        Returns
        -------
        x : np.ndarray
            Solution vector.

        Notes
        -----
        - Must be implemented by subclasses.
        - Subclasses may rely on `_A_id` and `_is_sparse` for matrix tracking.
        - Iterative solvers should implement convergence criteria.
        """
        pass

class DirectSolver(LinearSolver):
    """
    Direct linear solver for dense and sparse matrices with optional LU caching.

    This solver computes the exact solution to a linear system

        Ax = b

    using LU factorization:

    - Dense matrices (`np.ndarray`) use `scipy.linalg.lu_factor` and `lu_solve`.
    - Sparse matrices (`scipy.sparse.sparray`) use `scipy.sparse.linalg.splu`.

    LU factorization is cached to speed up repeated solves with the same matrix.

    Notes
    -----
    1. Sparse support: Sparse matrices are handled efficiently using `splu`. 
       Dense matrices are handled via `lu_factor`.
    2. Caching: Factorization is only recomputed if the input matrix changes.
    3. This solver is suitable for small to moderately sized systems.
       For very large systems, iterative solvers may be more efficient.
    4. Modifying the matrix `A` after factorization requires calling `reset()` or
       using a new `solve(A, b)` call to recompute LU.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[2, 1], [5, 7]])
    >>> b = np.array([11, 13])
    >>> solver = DirectSolver()
    >>> x = solver.solve(A, b)
    >>> print(x)
    [7.11111111, -3.22222222]

    >>> # Sparse example
    >>> from scipy.sparse import csr_array
    >>> row = np.array([0, 0, 1, 2, 2, 2])
    >>> col = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> A_sparse = csr_array((data, (row, col)), shape=(3, 3))
    >>> b = np.array([1, 2, 3])
    >>> x = solver.solve(A_sparse, b)
    """
    def __init__(self):
        """
        Initialize the DirectSolver.

        Notes
        -----
        - Internal LU factorization is set to None until `setup` is called.
        """
        self._lu : Any = None

    def setup(self, A: np.ndarray | sparray) -> None:
        """
        Precompute LU factorization for the given matrix.

        Parameters
        ----------
        A : np.ndarray or scipy.sparse.sparray
            Dense (`np.ndarray`) or sparse (`sparray`) matrix to factorize.

        Notes
        -----
        - Factorization is cached internally for repeated solves.
        - For sparse matrices, `scipy.sparse.linalg.splu` is used.
        - For dense matrices, `scipy.linalg.lu_factor` is used.
        """
        super().setup(A)
        if self._is_sparse:
            self._lu = sc.sparse.linalg.splu(A)
        else:
            self._lu = sc.linalg.lu_factor(A)

    def reset(self):
        """
        Invalidate cached LU factorization.

        Notes
        -----
        - Call this method if the matrix `A` has changed and a new factorization is required.
        """
        self._lu = None

    def solve(self, A: np.ndarray | sparray, b: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Solve the linear system Ax = b using LU factorization.

        Parameters
        ----------
        A : np.ndarray or scipy.sparse.sparray
            Dense or sparse system matrix.
        b : np.ndarray
            Right-hand side vector.
        **kwargs : Any
            Placeholder for solver-specific options (ignored in this solver).

        Returns
        -------
        x : np.ndarray
            Solution vector.

        Raises
        ------
        ValueError
            If `A` and `b` shapes are incompatible.
        """
        if self._lu is None or self._A_id != id(A):
            self.setup(A)

        if self._is_sparse:
            return self._lu.solve(b)
        else:
            return sc.linalg.lu_solve(self._lu, b)

class IterativeSolver(LinearSolver):
    """
    Base class for iterative linear solvers.

    This class provides a common interface and basic functionality for iterative 
    methods solving linear systems of the form:

        Ax = b

    Subclasses (e.g., Jacobi, Gauss-Seidel, SOR, Conjugate Gradient) should implement 
    the `solve` method. It also provides:

    - Storage of matrix sparsity information (`_is_sparse`) and identity (`_A_id`).
    - Initialization tracking (`_initialized`).
    - Optional caching and reset mechanism.
    - Default iteration error criterion (`scriterion`) using the infinity norm of differences.

    Important Notes
    ---------------
    1. Iterative solvers do not perform direct factorization.
    2. Subclasses should call `super().setup(A)` in their setup to store matrix info.
    3. Subclasses can override `reset` to invalidate cached state.
    4. Iterative solvers rely on convergence criteria such as residual norms or differences 
       between iterates, which can be customized by overriding `scriterion` or passing a 
       custom function to `solve`.

    Examples
    --------
    >>> class DummySolver(IterativeSolver):
    ...     def solve(self, A, b, tol=1e-6, maxiter=100, **kwargs):
    ...         return np.zeros_like(b)  # dummy implementation
    >>> solver = DummySolver()
    >>> x = solver.solve(np.array([[2,1],[1,3]]), np.array([1,2]))
    >>> print(x)
    [0 0]
    """
    def __init__(self):
        self._initialized = False

    def setup(self, A: np.ndarray | sparray) -> None:
        """
        Store matrix sparsity info and ID.

        Parameters
        ----------
        A : np.ndarray or scipy.sparse.sparray
            Dense or sparse matrix to store info for iterative use.

        Notes
        -----
        - This does not perform any factorization.
        - Sets `_initialized` to True.
        """
        super().setup(A)
        self._initialized = True

    def reset(self):
        """
        Invalidate any cached state.

        Notes
        -----
        - For basic iterative solvers (like Jacobi), there is typically no cached state.
        - Subclasses can override to clear solver-specific caches.
        """
        self._initialized = False

    def scriterion(self, x_old: np.ndarray, x_new: np.ndarray) -> float:
        """
        Compute a default convergence criterion for iterative solvers.

        Parameters
        ----------
        x_old : np.ndarray
            Previous iterate.
        x_new : np.ndarray
            Current iterate.

        Returns
        -------
        error : float
            Infinity norm of the difference between `x_new` and `x_old`.
        """
        return float(np.linalg.norm(x_new - x_old, ord=np.inf))

    @abstractmethod
    def solve(self, A: np.ndarray | sparray, b: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Abstract method to solve Ax = b iteratively.

        Parameters
        ----------
        A : np.ndarray or scipy.sparse.sparray
            Dense or sparse system matrix.
        b : np.ndarray
            Right-hand side vector.
        **kwargs : Any
            Solver-specific options (e.g., tolerance, maximum iterations, initial guess).

        Returns
        -------
        x : np.ndarray
            Approximate solution vector.

        Notes
        -----
        - Must be implemented by subclasses.
        - Can use `_initialized`, `_is_sparse`, and `_A_id` for matrix tracking and caching.
        """
        pass

class JacobiSolver(IterativeSolver):
    """
    Jacobi iterative solver for linear systems, supporting both dense and sparse matrices,
    with optional weighted (relaxed) iteration.  

    This solver computes an approximate solution to a linear system

        Ax = b

    using the Jacobi iteration:

        Plain Jacobi: x^{(k+1)} = D^{-1} (b - R x^{(k)}),  
        Weighted Jacobi: x^{(k+1)} = (1 - ω) x^{(k)} + ω D^{-1} (b - R x^{(k)}),  

    where D = diag(A), R = A - D, and ω (relfactor) is the relaxation factor.

    Important Notes
    ---------------
    1. Convergence:
       - Jacobi iteration converges if A is strictly diagonally dominant or symmetric positive definite.
       - Weighted Jacobi with 0 < ω < 1 is more stable; ω = 1 gives the standard Jacobi method.
       - ω > 1 can accelerate convergence in some cases but may cause divergence.

    2. Sparse vs dense:
       - Dense matrices (`np.ndarray`) use `np.diag` to extract the diagonal.
       - Sparse matrices (`scipy.sparse.sparray`) use `.diagonal()` and store the remainder as a sparse matrix.
       - This ensures memory efficiency for large sparse systems.

    3. Stopping criterion:
       - Iteration stops when the infinity norm of the difference between successive iterates
         is below `tol`, or when `maxiter` is reached.

    4. Logging:
       - Iteration count and current error are logged via the provided logger.
       - Final convergence or max-iteration warning is also logged.

    Parameters
    ----------
    None (class-level initialization). Use `solve` to provide matrix and vector.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 3]])
    >>> b = np.array([15, 10, 10])
    >>> solver = JacobiSolver()
    >>> x = solver.solve(A, b, tol=1e-6, maxiter=100, relfactor=0.8)
    >>> print(x)

    >>> # Sparse example
    >>> from scipy.sparse import csr_array
    >>> row = np.array([0, 0, 1, 2, 2, 2])
    >>> col = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> A_sparse = csr_array((data, (row, col)), shape=(3, 3))
    >>> b = np.array([1, 2, 3])
    >>> x = solver.solve(A_sparse, b, tol=1e-6)
    """
    def __init__(self):
        super().__init__()

    def solve(self, A: np.ndarray | sparray, b: np.ndarray, criterion = None, tol: float = 1e-8, maxiter: int = 1000, 
              relfactor: float = 0.3, x0: np.ndarray | None = None, **kwargs: Any) -> np.ndarray:
        """
        Solve the linear system Ax = b using Jacobi iteration.

        Parameters
        ----------
        A : np.ndarray or scipy.sparse.sparray
            Dense or sparse system matrix.
        b : np.ndarray
            Right-hand side vector.
        criterion : callable[[np.ndarray, np.ndarray], float], optional
            Function to compute iteration error between successive iterates.
            Defaults to infinity norm of difference (self.scriterion).
        tol : float, optional
            Convergence tolerance for the iteration error (default 1e-8).
        maxiter : int, optional
            Maximum number of iterations (default 1000).
        relfactor : float, optional
            Relaxation factor ω for weighted Jacobi (default 1.0, plain Jacobi).
            Must satisfy 0 < ω < 2 for convergence.
        x0 : np.ndarray, optional
            Initial guess for the solution (default: zero vector).
        **kwargs : Any
            Placeholder for solver-specific options (currently ignored).

        Returns
        -------
        x : np.ndarray
            Approximate solution vector of the linear system Ax = b.

        Raises
        ------
        ValueError
            If `A` and `b` shapes are incompatible.

        Notes
        -----
        - The returned solution is the latest iterate, which satisfies the convergence
          criterion if reached.
        - Iteration error is measured in the infinity norm by default.
        - Weighted Jacobi can improve stability or speed, but ω > 1 may diverge.
        """

        if not self._initialized or self._A_id != id(A):
            self.setup(A)

        if A.shape[0] != b.shape[0]: # type: ignore
            raise ValueError(f"Incompatible shapes: A has {A.shape[0]} rows but b has length {b.shape[0]}") # type: ignore

        if criterion is None:
            criterion = self.scriterion

        # Initial guess
        x = np.zeros_like(b) if x0 is None else x0.copy()

        # Extract diagonal and remainder
        if self._is_sparse:
            D = A.diagonal()
            R = A - sc.sparse.diags(D)
        else:
            D = np.diag(A)
            R = A - np.diagflat(D)

        logger.info("="*80)
        logger.info("[Jacobi Solver] Starting solver")
        logger.info("="*80)

        error: float = float("inf")

        # Jacobi iteration
        for iter in range(maxiter):
            x_new = (b - R @ x) / D
            x_new = (1 - relfactor) * x + relfactor * x_new  # apply weight
            error = criterion(x, x_new)
            logger.info(f"[Jacobi Solver] Iteration {iter + 1}: error = {error:.6e}")
            if error < tol:
                logger.info(f"[Jacobi Solver] Converged after {iter + 1} iterations with error = {error:.6e}")
                logger.info("="*80)
            else:
                x = x_new
        else:
            logger.warning(f"[Jacobi Solver] Reached max iterations ({maxiter}) with error = {error:.6e}")
        
        logger.info("[Jacobi Solver] Solver finished successfully")
        logger.info("="*80)

        return x

class CGSolver(IterativeSolver):
    """
    Conjugate Gradient (CG) iterative solver for symmetric positive definite (SPD) matrices.

    This solver wraps SciPy's ready-to-use `cg` function and supports optional preconditioning.

    Notes
    -----
    - Matrix must be SPD.
    - Prefer sparse matrices (`csr_array`, `csc_array`) for large FEM systems.
    - Optional preconditioner `M` can be a scipy.sparse LinearOperator or diagonal scaling.
    """

    def __init__(self):
        super().__init__()

    def solve(
        self,
        A: np.ndarray | sparray,
        b: np.ndarray,
        atol: float = 0.,
        rtol: float = 1e-5,
        maxiter: int = 1000,
        M: LinearOperator | None = None,
        x0: np.ndarray | None = None,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Solve Ax = b using Conjugate Gradient (CG) iteration.

        Parameters
        ----------
        A : np.ndarray or scipy.sparse.sparray
            Dense or sparse SPD system matrix.
        b : np.ndarray
            Right-hand side vector.
        atol, rtol : float, optional
            Parameters for the convergence test. For convergence, 
            norm(b - A @ x) <= max(rtol*norm(b), atol) should be satisfied. 
            The default is atol=0. and rtol=1e-5.
        maxiter : int, optional
            Maximum number of iterations (default 1000).
        M : LinearOperator or None, optional
            Preconditioner. Defaults to None (no preconditioning).
        x0 : np.ndarray or None, optional
            Initial guess for the solution (default: zero vector).
        **kwargs : Any
            Placeholder for future solver-specific options.

        Returns
        -------
        x : np.ndarray
            Approximate solution vector.

        Raises
        ------
        ValueError
            If shapes of A and b are incompatible.
        RuntimeError
            If CG fails to converge within maxiter.
        """
        if not self._initialized or self._A_id != id(A):
            self.setup(A)

        if A.shape[0] != b.shape[0]: # type: ignore
            raise ValueError(f"Incompatible shapes: A has {A.shape[0]} rows but b has length {b.shape[0]}") # type: ignore

        # Use zero vector as default initial guess
        if x0 is None:
            x0 = np.zeros_like(b)

        logger.info("="*80)
        logger.info("[CG Solver] Starting solver")
        logger.info(f"[CG Solver] maxiter={maxiter}, rtol={rtol}")
        logger.info("="*80)

        # Call SciPy's ready CG solver
        x, info = cg(A, b, x0 = x0, atol = atol, rtol = rtol, maxiter = maxiter, M = M)
        
        if info == 0:
            logger.info("[CG Solver] Converged successfully")
        elif info > 0:
            logger.warning(f"[CG Solver] Reached max iterations ({info}) without full convergence")
        else:
            raise RuntimeError(f"[CG Solver] Illegal input or breakdown: info={info}")

        logger.info("[CG Solver] Solver finished")
        logger.info("="*80)

        return x