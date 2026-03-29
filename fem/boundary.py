from __future__ import annotations
from typing import Callable, Optional, TYPE_CHECKING
from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import sparray, diags_array
from utils.logger import setup_logger
if TYPE_CHECKING:
    from fem.femspace import FEMSpace

logger = setup_logger(__name__, level='info')

# -------------- Boundary Conditions for Finite Element Method (FEM) ------------------
# The `BoundaryCondition` class is an abstract base class that defines the interface for
# boundary conditions in finite element problems. It provides a common structure for
# different types of boundary conditions (e.g., Dirichlet, Neumann, Robin) and requires
# subclasses to implement the `apply` method, which modifies the system matrix and
# right-hand side vector to enforce the boundary condition.
# --------------------------------------------------------------------------------------

class BoundaryCondition(ABC):
    """
    Abstract base class for boundary conditions in finite element problems.

    This class defines the interface for boundary conditions that modify the
    linear system

        K u = rhs

    arising from the discretization of a partial differential equation (PDE)
    using the finite element method.

    Subclasses implement specific types of boundary conditions such as:

        - Dirichlet (prescribed solution values)
        - Neumann (prescribed flux)
        - Robin (mixed boundary condition)

    Each boundary condition must implement the :meth:`apply` method, which
    modifies the system matrix and/or right-hand side vector so that the
    boundary condition is enforced.

    Parameters
    ----------
    femspace : FEMSpace
        Finite element space associated with the problem. The boundary
        condition operates on the mesh and degrees of freedom defined
        in this space.

    Attributes
    ----------
    femspace : FEMSpace
        Finite element space on which the boundary condition is defined.
    nnodes : int
        Total number of nodes (degrees of freedom) in the mesh.

    Examples
    --------
    Typical usage in a solver:

    >>> for bc in boundary_conditions:
    >>>     K, rhs = bc.apply(K, rhs)

    where ``boundary_conditions`` is a list containing instances of
    subclasses such as ``DirichletBC`` or ``NeumannBC``.
    """
    def __init__(self, femspace: "FEMSpace"):
        self.femspace = femspace
        self.nnodes = self.femspace.nnodes

    @abstractmethod
    def apply(self, K: np.ndarray | sparray, rhs: Optional[np.ndarray] = None) -> tuple[np.ndarray | sparray, Optional[np.ndarray]]:
        """
        Apply the boundary condition to the linear system.

        Implementations should modify the system matrix and/or the
        right-hand side vector so that the boundary condition is
        satisfied.

        Parameters
        ----------
        K : ndarray or sparray
            Global system matrix of shape (n, n).
        rhs : ndarray, optional, shape (n,)
            Right-hand side vector.

        Returns
        -------
        K : ndarray or sparray
            Modified system matrix.
        rhs : ndarray, optional, shape (n,)
            Modified right-hand side vector.
        """
        pass

class DirichletBC(BoundaryCondition):
    """
    Strong (nodal) Dirichlet boundary condition for finite element problems.

    This class enforces Dirichlet boundary conditions by modifying the global system
    matrix `K` and right-hand side vector `rhs` so that prescribed values at boundary
    nodes are exactly satisfied. It works efficiently for **dense and sparse matrices**.

    The classical strong imposition modifies the system as follows:

        K u = rhs  -->  K_mod u = rhs_mod

    where:
        - Rows and columns corresponding to Dirichlet nodes are zeroed
        - Ones are placed on the diagonal for Dirichlet rows
        - RHS is adjusted to incorporate Dirichlet values, preserving contributions
          to interior/Neumann nodes

    Parameters
    ----------
    femspace : FEMSpace
        Finite element space associated with the problem.
    g : Callable or dict
        Prescribed boundary function. Can be:
            - Static: `g(node)` → scalar
            - Time-dependent: `g(node, t)` → scalar
        If `g` is a dict, keys are treated as node indices and values as prescribed values.
    markers : list | np.ndarray, optional
        Segment markers identifying which boundary nodes the Dirichlet BC applies to.
        If `None` and g is dict, keys are treated as node indices. 
        If `None` and g is callable, BC is applied to all boundary nodes.
    time_steps : np.ndarray, shape (n_time_steps,), optional
        1D array of time points at which Dirichlet values are evaluated. 
        If `None`, BC is treated as static or it is assumed that `g` handles time-dependence internally.

    Attributes
    ----------
    markers : ndarray
        User-defined segment markers for this Dirichlet condition.
    dirichlet_nodes : np.ndarray of shape (n_dirichlet_nodes,)
        Global node indices where the Dirichlet BC is applied.
    dirichlet_nodes_coord : np.ndarray
        Coordinates of Dirichlet nodes.
        - For 1D: shape (n_dirichlet_nodes,)
        - For 2D: shape (n_dirichlet_nodes, 2)
    dirichlet_values : np.ndarray
        Precomputed Dirichlet values:
            - Shape `(n_dirichlet_nodes,)` for static BC
            - Shape `(n_dirichlet_nodes, n_time_steps)` for time-dependent BC
    _interior_mask : sparse diagonal matrix
        Mask with 1 for interior DOFs and 0 for Dirichlet DOFs (used for sparse K modification)
    _boundary_mask : sparse diagonal matrix
        Mask with 1 for Dirichlet DOFs and 0 for interior DOFs

    Notes
    -----
    - Only boundary DOFs should be assigned Dirichlet values.
    - Boundary nodes not included in `markers` are treated as free (Neumann or interior).
    - Sparse matrices are efficiently modified using diagonal masks to avoid loops.
    """
    def __init__(self, femspace: FEMSpace, g: Callable | dict, markers: Optional[list | np.ndarray] = None, time_steps: Optional[np.ndarray] = None):
        super().__init__(femspace)
        self.is_time_dependent = time_steps is not None or (isinstance(g, dict) and any(isinstance(v, (list, np.ndarray)) for v in g.values()))
        self.time_steps = time_steps
        if isinstance(g, Callable):
            self.g = self.vectorize(g)
            if markers is None: # Apply to all boundary nodes if no markers provided
                self.dirichlet_nodes = np.fromiter(self.femspace.mesh.boundary_nodes(), dtype=int)
            else:
                self.dirichlet_nodes = self.femspace.mesh.get_nodes(markers)
            self.dirichlet_values = self.g(self.femspace.mesh.vertices[self.dirichlet_nodes])
        else:
            if markers is not None:
                raise ValueError("When g is a dict, markers should be None. The keys of the dict are treated as node indices.")
            self.dirichlet_nodes = np.array(list(g.keys()), dtype = np.int64)
            self.dirichlet_values = np.asarray(list(g.values()), dtype = np.float64) # shape (n_dirichlet_nodes,) or (n_dirichlet_nodes, n_time_steps) if time-dependent
        # Prepare masks for sparse matrix modification
        mask = np.ones(self.nnodes, dtype = np.float64)
        mask[self.dirichlet_nodes] = 0.0
        bc_diag = np.zeros(self.nnodes, dtype = np.float64)
        bc_diag[self.dirichlet_nodes] = 1.0
        self._interior_mask = diags_array(mask)
        self._boundary_mask = diags_array(bc_diag)

    def vectorize(self, f: Callable):
        if self.femspace.dim == 1:
            if self.time_steps is None:
                return lambda x: f(x)  # x is (n,) → returns (n,)
            else:
                T = self.time_steps[None, :]  # shape (1, m)
                return lambda x: f(x[:, None], T)  # returns (n, m)
        elif self.femspace.dim == 2:
            if self.time_steps is None:
                return lambda x: f(x[:, 0], x[:, 1])  # returns (n,)
            else:
                T = self.time_steps[None, :]  # shape (1, m)
                return lambda x: f(x[:, 0][:, None], x[:, 1][:, None], T)  # returns (n, m)
        else:
            raise ValueError(f"Unsupported dimension: {self.femspace.dim}. Only 1D and 2D supported.")

    def update_dirichlet_values(self, new_values: Callable | dict):
        """
        This function allows updating the Dirichlet values at runtime, which is useful for problems 
        where K is constant but the BCs change over time (e.g., time-dependent problems or Schwarz iterations). 
        When new_values is a dict, it replaces the existing g and dirichlet_values. When new_values is a callable, 
        it updates dirichlet_values by evaluating the new function at the Dirichlet nodes. This allows for dynamic 
        BCs without needing to create a new DirichletBC instance. Note that when using a callable, the markers 
        and dirichlet_nodes remain unchanged, so the same set of nodes will be updated with new values.

        Parameters
        ----------
        new_values : Callable or dict
            New Dirichlet values. Can be:
                - Callable: g(node) → scalar or g(node, t) → scalar for time-dependent BCs
                - Dict: keys are node indices, values are prescribed values (static or time-dependent)
        
        Notes
        -----
        - When new_values is a dict, markers should be None and the keys are treated as node indices. The existing markers and dirichlet_nodes are ignored in this case.
        - When new_values is a callable, the existing markers and dirichlet_nodes are used to evaluate the new values at those nodes. The function is vectorized for efficiency.
        - This method allows for dynamic updating of Dirichlet values without modifying the structure of the boundary condition (i.e., which nodes are Dirichlet nodes remains the same).
        """
        if isinstance(new_values, dict):
            self.g = new_values
            self.dirichlet_nodes = np.fromiter(new_values.keys(), dtype=int)
            self.dirichlet_values = np.fromiter(new_values.values(), dtype=float)
        else:
            self.g = self.vectorize(new_values)
            self.dirichlet_values = self.g(self.femspace.mesh.vertices[self.dirichlet_nodes])

    def apply(self, K: np.ndarray | sparray, rhs: Optional[np.ndarray] = None, time_step: Optional[int] = None, copy: bool = True, modify_K: bool = True) -> tuple[np.ndarray | sparray, Optional[np.ndarray]]:
        """
        The function modifies the linear system
            K u = rhs
        so that prescribed Dirichlet values are enforced exactly at Dirichlet nodes.

        The modification follows the standard strong imposition procedure:
        1. Subtract contributions of fixed DOFs from the RHS (Schur complement)
        2. Zero rows and columns corresponding to Dirichlet DOFs
        3. Insert ones on the diagonal for Dirichlet rows
        4. Replace RHS values at Dirichlet DOFs with the prescribed values

        Dirichlet (strong) imposition via apply():
            - Prescribed values: u_D = g (known)
            - RHS modified for free DOFs (interior + Neumann):
                rhs[self.idofs] -= K[np.ix_(free_dofs, self.dirichlet_nodes)] @ self.dirichlet_values
            or equivalently (vectorized for speed):
                rhs -= K[:, self.dirichlet_nodes] @ self.dirichlet_values
            This is done **without loops or creating a new matrix**.
            - Dirichlet entries enforced exactly:
                rhs[self.dirichlet_nodes] = self.dirichlet_values
            - Matrix modification:
                K_II unchanged, K_ID = K_DI = 0, K_DD = I

        Final system after apply():

            [ K_II   0   K_IN ] [ u_I ]   [ f_I - K_ID g ]
            [  0     I     0  ] [ u_D ] = [      g        ]
            [ K_NI   0   K_NN ] [ u_N ]   [ f_N - K_ND g ]

        Matrix notation of RHS update (Schur complement):

            rhs_free = f_free - K_free,Dirichlet @ g
        where
            rhs_free = [ f_I; f_N ]               # interior + Neumann DOFs
            K_free,Dirichlet = [ K_ID; K_ND ]     # columns of K corresponding to Dirichlet DOFs
            g = prescribed Dirichlet values

        Neumann Boundary Conditions:
            - Prescribed flux q modifies the RHS directly:
                f_i += ∫_{Γ_N} φ_i * q ds
            - The stiffness matrix K is unchanged for Neumann nodes

        Two cases after apply():
        1. All boundary nodes are Dirichlet (pure Dirichlet):
            - K becomes block-diagonal: K_II for interior, I for Dirichlet
            - RHS incorporates Schur complement: f_I - K_ID g
            - Dirichlet values enforced exactly
        2. Mixed Dirichlet + Neumann:
            - Free DOFs = [u_I; u_N], Dirichlet rows/columns modified
            - RHS: Schur complement applied only for Dirichlet DOFs:
                f_I' = f_I - K_ID g
                f_N' = f_N - K_ND g
            - Neumann DOFs remain unknowns, Dirichlet DOFs are fixed
        After this procedure, Dirichlet values are enforced exactly, and the 
        RHS of free (interior + Neumann) DOFs is updated consistently to account 
        for the Dirichlet contribution.

        Parameters
        ----------
        K : np.ndarray or scipy.sparse.sparray
            Global system matrix of shape (n, n).
        rhs : np.ndarray of shape (n,), optional
            Right-hand side vector.
        time_step : int, optional, default=None
            Current time step index. This is used for time-dependent Dirichlet conditions.
        copy : bool, optional, default=True
            If True, create copies of K and rhs and apply modifications to them.
            If False, dense matrices are modified in-place; sparse matrices return a new matrix.
        modify_K : bool, optional, default=True
            If True, modify the matrix K to enforce Dirichlet conditions.
            If False, only modify the RHS vector to account for Dirichlet values, 
            without changing K. This can be useful when K is reused across time steps 
            and only the RHS changes due to time-dependent BCs.
            
        Returns
        -------
        K : np.ndarray or scipy.sparse.sparray
            Matrix with Dirichlet constraints applied.
        rhs : np.ndarray of shape (n,), optional
            Right-hand side vector with Dirichlet constraints applied.

        Notes
        -----
        - Matrix symmetry is preserved by zeroing both rows and columns
          associated with Dirichlet DOFs.
        - When ``copy=True``, the returned objects are new matrices/vectors.
        - When ``copy=False``, the input matrix and RHS are modified in-place.
        - Sparse matrices are modified using a diagonal masking approach
          to avoid costly format conversions.
        - The ``modify_K`` parameter controls whether the matrix K is modified
          to enforce Dirichlet conditions. If False, only the RHS is updated.
          This is useful for time-dependent problems where K is constant and only the RHS changes.
        """
        logger.debug(f"Applying nodal Dirichlet BC to {len(self.dirichlet_nodes)} nodes")
        
        # If not modifying K and rhs is None, there is nothing to modify, so we raise an error to avoid silent failures.
        if rhs is None and not modify_K:
            raise ValueError("If modify_K is False, rhs must be provided to account for Dirichlet contributions.")
        
        # Make copies if requested (for dense matrices, in-place modification is possible)
        if rhs is not None and copy:
            rhs = rhs.copy()

        # For sparse matrices, we return a new modified matrix. For dense matrices, we modify in-place if copy=False.
        if copy and not isinstance(K, sparray) and modify_K:
            K = K.copy()
        
        # Apply the Dirichlet BC using the appropriate method for dense or sparse matrices
        if isinstance(K, sparray):
            K = self._apply_sparse(K, rhs, time_step, modify_K)
        else:
            self._apply_dense(K, rhs, time_step, modify_K)
        logger.debug("Nodal Dirichlet BC applied")

        return K, rhs

    def _apply_dense(self, K: np.ndarray, rhs: Optional[np.ndarray] = None, time_step: Optional[int] = None, modify_K: bool = True):
        if rhs is not None: 
            mult = self.dirichlet_values[:, time_step] if time_step is not None else self.dirichlet_values
            rhs -= K[:, self.dirichlet_nodes] @ mult
            rhs[self.dirichlet_nodes] = mult
        if modify_K:
            K[self.dirichlet_nodes, :] = 0.0
            K[:, self.dirichlet_nodes] = 0.0
            K[self.dirichlet_nodes, self.dirichlet_nodes] = 1.0

    def _apply_sparse(self, K: sparray, rhs: Optional[np.ndarray] = None, time_step: Optional[int] = None, modify_K: bool = True) -> sparray:
        """
        Apply Dirichlet BCs to scipy sparse matrices.

        Instead of looping over boundary nodes and zeroing rows/columns one by one,
        we use the diagonal mask approach:
            K_mod = D @ K @ D + R
        where:
            D = diag(mask):  1 at interior DOFs, 0 at Dirichlet DOFs
            R = diag(bc):    1 at Dirichlet DOFs, 0 at interior DOFs
        D @ K @ D zeros both rows and columns of Dirichlet DOFs simultaneously,
        and R adds 1 on the diagonal at Dirichlet DOFs.

        This avoids format conversions (CSR <-> CSC) and Python loops,
        using only sparse matrix-matrix multiplies (optimized in C).
        """
        if rhs is not None:
            mult = self.dirichlet_values[:, time_step] if time_step is not None else self.dirichlet_values
            rhs -= K[:, self.dirichlet_nodes] @ mult # type: ignore
            rhs[self.dirichlet_nodes] = mult # type: ignore
        if modify_K:
            K = self._interior_mask @ K @ self._interior_mask + self._boundary_mask
        return K