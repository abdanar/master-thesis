import numpy as np
from fom.poisson import PoissonProblem
from fem.femspace import FEMSpace
from typing import Callable, Optional
from utils.logger import setup_logger
from utils.errornorms import ErrorNorms
from fem.linearsolver import LinearSolver, DirectSolver

logger = setup_logger(__name__, level = 'info')

class SchwarzProblem():
    def __init__(self, femspace: FEMSpace, f: Callable, g: Callable, n: int, overlap: int):
        """
        Initialize a Schwarz problem for solving the Poisson equation using Schwarz domain decomposition methods.

        Parameters
        ----------
        femspace : FEMSpace
            The finite element space representing the full computational domain.
        f : Callable
            The source function for the Poisson problem, defined as f(x) for 1D or f(x, y) for 2D problems.
        g : Callable
            The Dirichlet boundary condition function.
            - It should be defined as g(x) for 1D or g(x, y) for 2D problems.
        n : int
            Number of subdomains to decompose the mesh into.
        overlap : int
            Number of layers added to a non-overlapping decomposition to create overlap.
        """
        self.femspace = femspace
        self.f = f
        self.g = g
        self.n = n
        self.overlap = overlap
        logger.info(f"[Schwarz] Decomposing mesh into {n} subdomains with overlap of {overlap} layer(s)...")  
        self.subdomains, self.ltog, self.gtol, self.maps, _ = self.femspace.mesh.decompose(n = n, overlap = overlap) # speed up decompose!
        logger.info(f"[Schwarz] Mesh decomposition completed. Number of subdomains: {len(self.subdomains)}")
        self.nspace = self.femspace.nnodes
        verts = self.femspace.mesh.vertices
        boundary_nodes = np.array(list(self.femspace.mesh.boundary_nodes()), dtype = np.int64)
        boundary_nodes_coord = verts[boundary_nodes]
        logger.info(f"[Schwarz] Evaluating Dirichlet boundary conditions at {len(boundary_nodes)} boundary nodes...")
        if self.femspace.dim == 1:
            values = g(boundary_nodes_coord)
        elif self.femspace.dim == 2:
            values = g(boundary_nodes_coord[:, 0], boundary_nodes_coord[:, 1])
        else:
            raise ValueError("Unsupported dimension. Only 1D and 2D problems are supported.")
        logger.info(f"[Schwarz] Dirichlet boundary conditions evaluated successfully.")
        self.dirichlet = dict(zip(boundary_nodes, values))
        self.error_history = []
    
    def construct_dirichlet_bc(self, method: str, domainID: int, data: dict) -> dict:
        """
        Construct Dirichlet boundary values for a specific subdomain, accounting for shared nodes.

        This function computes the Dirichlet boundary values for the nodes of a given subdomain.
        For nodes shared by multiple subdomains, the boundary value is computed considering the method:
        - For 'AS' (Additive Schwarz), the value is the sum of contributions from all subdomains sharing that node.
        - For 'RAS' (Restricted Additive Schwarz), the value is the average of contributions from all subdomains sharing that node.
        However, if any of the shared entries corresponds to subdomain 0 (the original global domain), 
        the Dirichlet value from `self.dirichlet` is used directly.             

        Parameters
        ----------
        method : str
            The Schwarz method to use ('AS' or 'RAS').
        domainID : int
            The ID of the subdomain for which Dirichlet data is constructed.
        data : dict
            Dictionary containing subdomain solutions. Keys are subdomain IDs (starting from 1),
            and values are arrays of shape `(local_dofs,)` representing the local solution 
            for each subdomain.

        Returns
        -------
        dirichlet_bc : dict
            A dictionary mapping local boundary node indices of `domainID` to their 
            Dirichlet boundary values.
        """    
        dirichlet_bc = {}
        for dindex, dlist in self.maps[domainID].items():
            share = len(dlist) # number of subdomains that shares dindex node
            # Check if any entry uses Dirichlet value
            dirichlet_entry = next(((i, j) for (i, j) in dlist if i == 0), None)
            if dirichlet_entry is not None:
                # If any i==0, assign Dirichlet value directly
                _, j = dirichlet_entry
                dirichlet_bc[dindex] = self.dirichlet[j]
            elif share > 1:
                # Average over all subdomains sharing this node
                val = sum(data[i][j] for (i, j) in dlist) 
                if method == 'AS':
                    dirichlet_bc[dindex] = val
                elif method == 'RAS':
                    dirichlet_bc[dindex] = val/share
                else:
                    raise ValueError(f"Invalid method '{method}'. Must be 'RAS' or 'AS'.")
            else:
                # Only one subdomain, take its value
                i, j = dlist[0]
                dirichlet_bc[dindex] = data[i][j]
        return dirichlet_bc
    
    def combine(self, method: str, data: dict) -> np.ndarray:
        """
        Assemble a global solution from subdomain solutions.

        Global solution is constructed by mapping local subdomain solutions 
        to the global domain using `self.ltog`. The assembly process depends 
        on the chosen method:

        - For RAS (`method = 'RAS'`), contributions from all subdomains are averaged for 
          shared DOFs, resulting in a weighted average at overlaps. 

        - For AS (`method = 'AS'`), contributions from all subdomains are added together 
          for shared DOFs, resulting in a sum of contributions at overlaps. This can lead to a 
          more accurate solution on overlaps but may require relaxation (omega < 1) for stability 
          when multiple overlapping subdomains are present.

        An important note is that global construction is performed by combining local solutions interior 
        to the subdomains (only interior nodes of each subdomain), while the Dirichlet boundary conditions 
        are applied directly to the global solution. This means that the values at the boundary nodes of 
        the subdomains are not used in the assembly process, but rather the Dirichlet values from `self.g` 
        are applied to the global solution at the corresponding indices. This approach ensures that the 
        boundary conditions are correctly enforced in the global solution, while the interior values 
        from the subdomains are combined according to the chosen method.
        
        Parameters
        ----------
        method : str
            The Schwarz method to use for combining subdomain solutions ('RAS' or 'AS').
        data : dict
            Dictionary containing subdomain solutions. Keys are subdomain IDs (starting from 1),
            and values are arrays of shape `(local_dofs,)` representing the local solution 
            for each subdomain.

        Returns
        -------
        global_solution : ndarray, shape (nspace,)
            Global solution assembled from subdomain solutions.

        Raises
        ------
        ValueError
            If `method` is not 'RAS' or 'AS'.
        """
        # Initialize global solution array
        global_solution = np.zeros(self.nspace)
        
        # Apply Dirichlet boundary conditions
        for idx, values in self.dirichlet.items():
            global_solution[idx] = values

        # The dictionary that contains arrays for local dof to global dof mappings for each subdomains with keys to be domainID
        local_to_global = self.ltog

        # Assemble global solution from subdomain solutions based on the chosen method
        if method == 'RAS':
            count = np.zeros(self.nspace)  
            for subdomain in self.subdomains:
                i = subdomain.domainID
                bdindices = subdomain.boundary_nodes()
                for local_index, global_index in enumerate(local_to_global[i]):
                    if local_index not in bdindices:
                        global_solution[global_index] += data[i][local_index]
                        count[global_index] += 1
            for ix in range(self.nspace):
                if count[ix] > 0:
                    global_solution[ix] /= count[ix]
        elif method == 'AS':
            for subdomain in self.subdomains:
                i = subdomain.domainID
                bdindices = subdomain.boundary_nodes()
                for local_index, global_index in enumerate(local_to_global[i]):
                    if local_index not in bdindices:
                        global_solution[global_index] += data[i][local_index] # sum contributions from all subdomains, to have perfect solution on overlaps remove + sign
        else:
            raise ValueError(f"Invalid method '{method}'. Must be 'RAS' or 'AS'.")
        return global_solution
    
    @staticmethod
    def boundary_criterion(data_old: dict, data_new: dict) -> float:
        """
        Evaluate the convergence criterion on subdomain boundaries.

        Parameters
        ----------
        data_old : dict
            Dictionary containing subdomain solutions from the previous iteration.
            Keys are subdomain IDs (starting from 1), and values are arrays of shape
            `(local_dofs,)` representing the local solution for each subdomain.
        data_new : dict
            Dictionary containing subdomain solutions from the current iteration.
            Same format as `data_old`.

        Returns
        -------
        float
            Maximum difference on boundary nodes between the old and new solutions
            across all subdomains and all time steps.
        """
        max_diff = 0.0
        for subdomain_id in data_old:
            old_sol = data_old[subdomain_id]
            new_sol = data_new[subdomain_id]
            diff = np.abs(new_sol - old_sol).max()
            if diff > max_diff:
                max_diff = diff
        return max_diff
    
    def initial_data(self) -> dict:
        """
        Initialize the local solution data for each subdomain.

        This function creates a dictionary mapping subdomain IDs to their initialized local 
        solution arrays. The initialization depends on the type of `self.g`:
        - If `self.g` is a dictionary, the local solution arrays are initialized to zero, and the 
          Dirichlet values from `self.g` are applied to the corresponding indices.
        - If `self.g` is a function, the local solution arrays are initialized to zero, and the 
          Dirichlet values are evaluated at the boundary nodes of each subdomain and applied accordingly.     

        Returns
        -------
        idata : dict
            A dictionary mapping subdomain IDs to their initialized local solution arrays.
        """
        # The dictionary that contains dictionary for global dof to local dof mappings for each subdomains with keys to be domainID
        global_to_local = self.gtol

        # Construct initial data for each subdomain
        idata = {}
        for subdomain in self.subdomains:
            global_indices = global_to_local[subdomain.domainID]
            data = np.zeros(subdomain.nnodes())
            for idx, values in self.dirichlet.items():
                if idx in global_indices:
                    data[global_indices[idx]] = values
            idata[subdomain.domainID] = data
        return idata

    def solve(self, lift: str = 'nodal', method: str = 'RAS', omega: float = 1.0, 
              solver: LinearSolver = DirectSolver(), maxiter: int = 100, tol: float = 1e-3, criterion: Callable = boundary_criterion, 
              history: bool = False, norm: str = 'l2', uh: Optional[np.ndarray] = None, exact: Optional[Callable] = None) -> np.ndarray:
        """
        Solve the Poisson problem using the Overlapping Schwarz method.

        Parameters
        ----------
        lift : str
            Type of lifting function used to solve the local problems. Options include 'harmonic' for harmonic lifting and 'nodal' for nodal lifting.
        method : str, default = 'RAS'
            Schwarz method, either 'RAS' (Restricted Additive Schwarz) or 'AS' (Additive Schwarz).
        omega : float, optional
            Relaxation parameter for the Schwarz iteration. The global iterate is updated as
                u^{k+1} = (1 - omega) u^k + omega * u_tilde^{k+1}.
            Values 0 < omega <= 1 are allowed. Using omega < 1 stabilizes the additive Schwarz method for multiple overlapping subdomains.
            Default is 1.0 (no relaxation).
        solver : LinearSolver, optional
            Linear solver to use for solving the linear system. Must be an instance of a class that 
            inherits from `LinearSolver`. Default is `DirectSolver()`.
        maxiter : int, default = 100
            Maximum number of Schwarz iterations.
        tol : float, default = 1e-3
            Convergence tolerance for the stopping criterion.
        criterion : Callable, default = boundary_criterion
            Convergence criterion function. It should accept the old and new subdomain solutions and return a float representing the error.
        history : bool, default = False
            Whether to store the error history at each iteration for later analysis.
        norm : str, default = 'l2'
            Norm type to use for error analysis if `history` is True. Options include 'l2', 'linf', etc. (see `ErrorNorms` class for supported norms).
        uh : ndarray, optional
            The finite element solution obtained from solving the global problem on the full mesh. 
            Used for error analysis if `history` is True.
        exact : Callable, optional
            The exact solution function for the Poisson problem. Used for error analysis if `history` is True.
        
        Returns
        -------
        np.ndarray, shape (nspace,)
             The computed global solution vector at the FEM nodes, assembled from the subdomain solutions using the specified Schwarz method and relaxation.
        """
        logger.info("="*80)
        logger.info("[Schwarz] Starting solver")
        logger.info(
            f"dim: {self.femspace.dim}D | "
            f"subdomains: {len(self.subdomains)} | "
            f"method: {method} | "
            f"omega: {omega:.2f} | "
            f"maxiter: {maxiter} | "
            f"tol: {tol:.2e}")
        logger.info("="*80)

        error: float = float("inf")
        data = self.initial_data()

        # Create Poisson problems for each subdomain (not solved yet, just initialized)
        subproblems = {subdomain.domainID: PoissonProblem(femspace = FEMSpace(mesh = subdomain, domain = self.femspace.domain, space = self.femspace.space, degree = self.femspace.degree),
            f = self.f, g = self.construct_dirichlet_bc(method = method, domainID = subdomain.domainID, data = data)) for subdomain in self.subdomains}

        for iter in range(maxiter):
            new_data = {}
            for subdomain in self.subdomains:
                domainid = subdomain.domainID
                dirichlet_bc = self.construct_dirichlet_bc(method = method, domainID = domainid, data = data) if iter > 0 else None
                new_data[domainid] = subproblems[domainid].solve(lift = lift, solver = solver, g_new = dirichlet_bc)

            error = criterion(data, new_data)
            logger.info(f"[Schwarz] Iteration {iter + 1}: error = {error:.6e}")

            if error < tol:
                logger.info(f"[Schwarz] Converged after {iter + 1} iterations with error = {error:.6e}")
                logger.info("="*80)
                break
            else:
                if omega == 1.0:
                    data = new_data
                else:
                    for i in data:
                        data[i] = (1 - omega) * data[i] + omega * new_data[i]

            if history: # store error history
                schwarz_sol = self.combine(method = method, data = data)
                est = ErrorNorms(femspace = self.femspace, u1 = schwarz_sol, u2 = uh, u_exact = exact)
                error_norm = est.compute(norm)
                logger.info(f"[Schwarz] Iteration {iter + 1}: error norm ({norm}) = {error_norm:.6e}")
                self.error_history.append(error_norm)
        else:
            logger.warning(f"[Schwarz] Reached max iterations ({maxiter}) with error = {error:.6e}")
        
        logger.info("[Schwarz] Solver finished successfully")
        logger.info("="*80)

        return self.combine(method = method, data = data)